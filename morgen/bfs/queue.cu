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


/* texture memory */
texture<int> tex_row_offsets;
texture<int> tex_column_indices;


template<typename VertexId,
         typename SizeT, 
         typename Value,
         bool ORDERED>
__global__ void
BFSKernel_queue_thread_map(
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

        //SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeFirst = tex1Dfetch(tex_row_offsets, outNode);

        //SizeT outEdgeLast = row_offsets[outNode+1];
        SizeT outEdgeLast = tex1Dfetch(tex_row_offsets, outNode+1);

        // serial expansion
        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            //VertexId inNode = column_indices[edge];
            VertexId inNode = tex1Dfetch(tex_column_indices, edge);

            Value level = curLevel + 1;

            if (ORDERED) {
                int old = atomicExch( (int*)&visited[inNode], 1 );
                if (old == 0) { 
                    levels[inNode] = level;
                    SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                    worksetTo[pos] = inNode;
                }
            } else {
                if (levels[inNode] > level) {
                    levels[inNode] = level;     // relax
                    SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                    worksetTo[pos] = inNode;
                }
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
         typename Value,
         bool ORDERED>
__global__ void
BFSKernel_queue_group_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    VertexId  *worksetTo,
    SizeT     *sizeTo,
    Value     *levels,
    Value     curLevel,
    int       *visited,
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

        //if (g % 1024 == 0 && group_offset == 0)
        //    printf("I am group %d, size: %d, my next: %d\n", g, *sizeFrom, g+group_per_block * gridDim.x);


        // First thread in the group do this job read out info 
        // from global mem to local mem
        if (group_offset == 0) {

            VertexId outNode = worksetFrom[g];
            edge_first[group_id % group_per_block] = row_offsets[outNode];
            edge_last[group_id % group_per_block] = row_offsets[outNode+1];
        }

        __syncthreads();
    
        // in case the neighbor number > warp size
        for (SizeT edge = edge_first[group_id % group_per_block] + group_offset;
             edge < edge_last[group_id % group_per_block];
             edge += group_size)
        {
            
            VertexId inNode = column_indices[edge];

            Value level = curLevel + 1;

            if (ORDERED) {
                int old = atomicExch( (int*)&visited[inNode], 1 );
                if (old == 0) { 
                    levels[inNode] = level;
                    SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                    worksetTo[pos] = inNode;
                } 
            } else {
                if (levels[inNode] > level) {
                    levels[inNode] = level;
                    SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                    worksetTo[pos] = inNode;
                }
            } // ordered
        }
    }
}



template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_queue(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    bool instrument,
    int block_size,
    bool warp_mapped,
    int group_size,
    bool unordered)
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
    int blockNum;
    int mapping_factor = (warp_mapped) ? group_size : 1; 
    int group_per_block = block_size / group_size;

    printf("GPU queued bfs starts... \n");  
    if (instrument) printf("level\tfrontier_size\tblock_num\ttime\n");


    /*
    * bind the graph in texture memory(1D)
    */
    if (util::handleError(cudaBindTexture(0, tex_column_indices, g.d_column_indices, sizeof(VertexId) * g.m), 
        "CsrGraph: bindTexture(d_column_indices) failed", __FILE__, __LINE__)) exit(1);        

    if (util::handleError(cudaBindTexture(0, tex_row_offsets, g.d_row_offsets, sizeof(SizeT) * (g.n + 1)), 
        "CsrGraph: bindTexture(d_row_offsets) failed", __FILE__, __LINE__)) exit(1);
        
    printf("Done texture memory binding.\n");


    while (worksetSize > 0) {

        // kick off timer first
        util::GpuTimer gpu_timer;
        gpu_timer.start();

        lastWorksetSize = worksetSize;

        // spawn minimal(but enough) software blocks to cover the workset
        blockNum = (worksetSize * mapping_factor % block_size == 0 ? 
            worksetSize * mapping_factor / block_size :
            worksetSize * mapping_factor / block_size + 1);
        
        // safe belt: grid width has a limit of 65535
        if (blockNum > 65535) blockNum = 65535;


        if (warp_mapped) {
            if (unordered) {
                BFSKernel_queue_group_map<VertexId, SizeT, Value, false><<<blockNum, block_size>>>(
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
                    group_per_block);

            } else {
                // unorderd
                BFSKernel_queue_group_map<VertexId, SizeT, Value, true><<<blockNum, block_size>>>(
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
                    group_per_block);
            }

        } else { // thread map

            if (unordered) {
                BFSKernel_queue_thread_map<VertexId, SizeT, Value, false><<<blockNum, block_size>>>(
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
                // unordered
                BFSKernel_queue_thread_map<VertexId, SizeT, Value, true><<<blockNum, block_size>>>(
                    g.d_row_offsets,                                        
                    g.d_column_indices,
                    workset[selector].d_elems,
                    workset[selector].d_sizep,
                    workset[selector ^ 1].d_elems,
                    workset[selector ^ 1].d_sizep,
                    levels.d_elems,
                    curLevel,     
                    visited.d_elems);
            }
        }


        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;
        worksetSize = workset[selector ^ 1].size();

        gpu_timer.stop();

        if (instrument) printf("%d\t%d\t%d\t%f\n", curLevel, lastWorksetSize, blockNum, gpu_timer.elapsedMillis());        


        total_milllis += gpu_timer.elapsedMillis();
        accumulatedBlocks += blockNum;
        curLevel += 1;
        selector = selector ^ 1;

    } // endwhile


    
    printf("GPU queued bfs terminates\n");  
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