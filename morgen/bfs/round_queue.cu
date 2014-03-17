/*
 *   The breadth-first search algorithm
 *
 *   Copyright (C) 2013-2014 by
 *   Cheng Yichao        onesuperclark@gmail.com
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 */


#pragma once


#include <morgen/utils/macros.cuh>
#include <morgen/utils/timing.cuh>
#include <morgen/utils/list.cuh>
#include <morgen/workset/queue.cuh>
#include <cuda_runtime_api.h>





namespace morgen {

namespace bfs {


template<typename VertexId,
         typename SizeT, 
         typename Value>
__global__ void
BFSKernel_round_queue_thread_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    VertexId  *worksetTo,
    SizeT     *sizeTo,
    Value     *levels,
    Value     curLevel,
    int       *visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // writing to an empty buffer
    if (tid == 0) *sizeTo = 0;
    __syncthreads();

    if (tid < *sizeFrom) {
        
        // read the who-am-I info from the workset
        VertexId outNode = worksetFrom[tid];
        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];

        // serial expansion
        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            VertexId inNode = column_indices[edge];

            int old = atomicExch( (int*)&visited[inNode], 1 );
            if (old == 0) { 
                levels[inNode] = curLevel + 1;
                SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                worksetTo[pos] = inNode;
            }

        }   
    }
}



/**
 * Each vertex(u) in worksetFrom is assigned with a group of threads.
 * Then each thead within a group processes one of u's neigbors
 * at a time. All threads process vertices in SIMD manner.
 *
 * Assume GROUP_S = 32
 * If u has a neigbor number more than 32, each thead within a group will 
 * iterate over them stridedly. e.g. thread 1 will process 1st, 33th, 65th... 
 * vertex in the neighbor list, thread 2 will process 2nd, 34th, 66th...
 */
template<typename VertexId, 
         typename SizeT, 
         typename Value>
__global__ void
BFSKernel_round_queue_group_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    VertexId  *worksetTo,
    SizeT     *sizeTo,
    Value     *levels,
    Value     curLevel,
    int       *visited,
    int       mask,
    int       group_size,
    int       group_per_block)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int group_offset = tid % group_size;
    int group_id     = tid / group_size;

    // writing to an empty buffer
    if (tid == 0) *sizeTo = 0;

    __syncthreads();


    // Each group has use a variable to record the total work
    // amount(neigbors) that belongs to that group
    // groups/block = thread per block / group size
    // The size is allocated dynamically
    volatile __shared__ SizeT edge_first[256];
    volatile __shared__ SizeT edge_last[256];


    // Since the workset can easily exceed 65536, we just let grouped-threads
    // iterate over a large workset
    for (int g = group_id; g < *sizeFrom; g += group_per_block * gridDim.x) {

        // First thread in the group do this job read out info 
        // from global mem to local mem
        if (group_offset == 0) {
            VertexId outNode = worksetFrom[g];
            edge_first[group_id % group_per_block] = row_offsets[outNode];
            edge_last[group_id % group_per_block] = row_offsets[outNode+1];
        }

        __syncthreads();
    
        SizeT edgeFirst = edge_first[group_id % group_per_block];
        SizeT edgeLast = edge_last[group_id % group_per_block];

        SizeT edge_num = edgeLast - edgeFirst;

        if (mask & edge_num != 0) {

            SizeT skip_edges = (~mask) & edge_num;

            for (int e = edgeFirst + skip_edges + group_offset; e < edgeLast; e += group_size) {
                
                VertexId inNode = column_indices[e];
                int old = atomicExch( (int*)&visited[inNode], 1 );
                if (old == 0) { 
                    levels[inNode] = curLevel + 1;
                    SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                    worksetTo[pos] = inNode;
                } 

            }

        }

    }
}



template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_round_queue(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    const util::Stats<VertexId, SizeT, Value> &stats,
    bool instrument,
    int block_size)
{

    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Queue<VertexId, SizeT>  workset[] = {
        workset::Queue<VertexId, SizeT>(g.n),
        workset::Queue<VertexId, SizeT>(g.n),
    };

    // use to select between two worksets
    // src:  workset[selector]
    // dest: workset[selector ^ 1]
    int selector = 0;

    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);

    // traverse from source node
    workset[0].init(source);   
    levels.set(source, 0);
    visited.set(source, 1);
    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 0;
    Value curLevel = 0;
    float total_milllis = 0.0;
    int accumulatedBlocks = 0;

    // kernel configuration
    int blockNum = 16;

    printf("GPU roundly-queued bfs starts... \n");  
    if (instrument) printf("level\tfrontier_size\tblock_num\ttime\n");


    int max_outdegree_log = stats.outDegreeMax;


    while (worksetSize > 0) {


        float level_millis = 0.0;

        lastWorksetSize = worksetSize;

        for (int i = 0; i < max_outdegree_log; i++) {

            util::GpuTimer gpu_timer;
            gpu_timer.start();


            int group_size = 0;
            switch (i) {
                case 0: group_size = 1; break;
                case 1: group_size = 2; break;
                case 2: group_size = 4; break;
                case 3: group_size = 8; break;
                case 4: group_size = 16; break;
                case 5: group_size = 32; break;
                default: group_size = 32;
            }

            int group_per_block = block_size / group_size;


            // spawn minimal(but enough) software blocks to cover the workset
            blockNum = (worksetSize * group_size % block_size == 0 ? 
                worksetSize * group_size / block_size :
                worksetSize * group_size / block_size + 1);

            // safe belt: grid width has a limit of 65535
            if (blockNum > 65535) blockNum = 65535;

            if (group_size == 1) {

                BFSKernel_round_queue_thread_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                    g.d_row_offsets,                                        
                    g.d_column_indices,
                    workset[selector].d_elems,
                    workset[selector].d_sizep,
                    workset[selector ^ 1].d_elems,
                    workset[selector ^ 1].d_sizep,
                    levels.d_elems,
                    curLevel,     
                    visited.d_elems);

            } else {
                BFSKernel_round_queue_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                    g.d_row_offsets,
                    g.d_column_indices,
                    workset[selector].d_elems,
                    workset[selector].d_sizep,
                    workset[selector ^ 1].d_elems,
                    workset[selector ^ 1].d_sizep,
                    levels.d_elems,
                    curLevel,     
                    visited.d_elems,
                    group_size,
                    group_size,
                    group_per_block);


            } // if else

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;
            
            gpu_timer.stop();
            level_millis += gpu_timer.elapsedMillis();
            if (instrument) printf("[round]%d\t%f\n", i, gpu_timer.elapsedMillis());

        } // for


        worksetSize = workset[selector ^ 1].size();
        if (instrument) printf("%d\t%d\t%d\t%f\n", curLevel, lastWorksetSize, blockNum, level_millis);        

        total_milllis += level_millis;
        accumulatedBlocks += blockNum;
        curLevel += 1;
        selector = selector ^ 1;

    } // endwhile


    
    printf("GPU roundly-queued bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_milllis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_milllis / 1000.0, billion_edges_per_second);
    printf("Accumulated Blocks: \t%d\n", accumulatedBlocks);

    levels.print_log();

    levels.del();
    visited.del();
    workset[0].del();
    workset[1].del();
    
}


} // BFS

} // Morgen