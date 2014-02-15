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
#include <morgen/utils/var.cuh>
#include <morgen/utils/stats.cuh>


#include <cuda_runtime_api.h>




namespace morgen {

namespace bfs {

/**
 * each thread wakeup and check if activated[tid] == 1
 * using update[] to mark unvisited vertices in this round
 */
template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_expand_single_thread(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *activated,
    Value     curLevel,
    Value     *levels,
    int       *visited,
    int       *update)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < max_size) {

        if (activated[tid] == 1) {


            SizeT outEdgeFirst = row_offsets[tid];
            SizeT outEdgeLast = row_offsets[tid+1];

            SizeT edge_num = outEdgeLast - outEdgeFirst;

            if (1 & edge_num != 0) { 


                if (edge_num == 1) activated[tid] = 0;

                // each threads only expand an edge 
                VertexId inNode = column_indices[outEdgeFirst];  

                if (visited[inNode] == 0) {
                    levels[inNode] = curLevel + 1;
                    update[inNode] = 1;

                }



            }
        }
    }
}


/**
 * each thread wakeup and check if activated[tid] == 1
 * using update[] to mark unvisited vertices in this round
 */
template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_expand_group(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *activated,
    Value     curLevel,
    Value     *levels,
    int       *visited,
    int       *update,
    int       mask,
    int       group_size,
    int       group_per_block)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    int group_offset = tid % group_size;
    int group_id     = tid / group_size;


    // Each group has use a variable to record the total work
    // amount(neigbors) that belongs to that group
    // groups/block = thread per block / group size
    // The size is determined dynamically
    volatile __shared__ SizeT edge_first[256];
    volatile __shared__ SizeT edge_last[256];

    for (int g = group_id; g < max_size; g += group_per_block * gridDim.x) {


        if (activated[g] == 1) {  // neglect the inactivated node



            // #0 thread in group is in charge of fetching 
            if (group_offset == 0) {
                edge_first[group_id % group_per_block] = row_offsets[g];
                edge_last[group_id % group_per_block] = row_offsets[g+1];
            }

            __syncthreads();

            SizeT edgeFirst = edge_first[group_id % group_per_block];
            SizeT edgeLast = edge_last[group_id % group_per_block];

            SizeT edge_num = edgeLast - edgeFirst;


            if ((mask & edge_num) != 0) {  // the certain round is activated


                SizeT skip_edges = (mask-1) & edge_num;  // mask off the higher bits

                if (skip_edges + mask == edgeLast) {  // which means its the last round of expansion
                    if (group_offset == 0) {
                        activated[g] = 0;
                    }
                }

                for (int e = edgeFirst + skip_edges + group_offset; e < edgeLast; e += group_size) {
                    VertexId inNode = column_indices[e];
                    levels[inNode] = curLevel + 1;
                    update[inNode] = 1;


                }
            }
        }
    }
}

/**
 * use update[] to mask activated[]
 */
template<typename VertexId, typename SizeT>
__global__ void
BFSKernel_update_round(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *activated,
    int       *visited,
    int       *update,
    int       *terminate)
{
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < max_size) {

        if (update[tid] == 1) {
            activated[tid] = 1;     
            update[tid] = 0;     // clear after activating
            visited[tid] = 1;   
            // as long as one thread try to set it false
            // the while loop will not be terminated 
            *terminate = 0; 

        }
    }
}


template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_round_bitmask(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    const util::Stats<VertexId, SizeT, Value> &stats,
    bool instrument,
    int block_size)
{

    // use a list to represent bitmask
    util::List<int, SizeT> activated(g.n);
    util::List<int, SizeT> update(g.n);
    activated.all_to(0);
    update.all_to(0);

    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    // visitation
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);

    // set up a flag, initially set
    util::Var<int> terminate;
    terminate.set(0);

    // traverse from source node
    activated.set(source, 1);
    levels.set(source, 0);
    visited.set(source, 1);
    Value curLevel = 0;


    int max_outdegree_log = stats.outDegreeMax;

    printf("GPU round bitmasked bfs starts... \n");   
    if (instrument) printf("level\ttime\n");

    float total_milllis = 0.0;


    // loop as long as the flag is set
    while (terminate.getVal() == 0) {

        // set true at first, if no vertex has been expanded
        // the while loop will be terminated
        terminate.set(1);

        float level_millis = 0.0;

        for (int i = 0; i < max_outdegree_log; i++) {
            // kick off timer first
            util::GpuTimer gpu_timer;
            gpu_timer.start();

            int group_size = 0;
            int mask = 0;
            switch (i) {
                case 0: group_size = 1; mask = 1; break;
                case 1: group_size = 2; mask = 2; break;
                case 2: group_size = 4; mask = 4; break;
                case 3: group_size = 8; mask = 8; break;
                case 4: group_size = 16; mask = 16; break;
                case 5: group_size = 32; mask = 32; break;
                case 6: group_size = 32; mask = 64; break;
                case 7: group_size = 32; mask = 128; break;
                case 8: group_size = 32; mask = 256; break;
                case 9: group_size = 32; mask = 512; break;
                case 10: group_size = 32; mask = 1024; break;
                case 11: group_size = 32; mask = 2048; break;
                case 12: group_size = 32; mask = 4096; break;
                case 13: group_size = 32; mask = 8192; break;
                case 14: group_size = 32; mask = 16384; break;
                case 15: group_size = 32; mask = 32768; break;
                default: fprintf(stderr, "out of control!!\n"); return;
            }


            // will be used in the kernel
            int group_per_block = block_size / group_size;


            // spawn as many threads as the vertices in the graph
            int blockNum = (g.n * group_size % block_size == 0 ? 
                g.n * group_size / block_size :
                g.n * group_size / block_size + 1);

            // safe belt: grid width has a limit of 65535
            if (blockNum > 65535) blockNum = 65535;

            if (group_size == 1) {
                BFSKernel_expand_single_thread<<<blockNum, block_size>>>(
                    g.n,
                    g.d_row_offsets,
                    g.d_column_indices,
                    activated.d_elems,
                    curLevel,
                    levels.d_elems,             
                    visited.d_elems,
                    update.d_elems);
        
            } else {
                BFSKernel_expand_group<<<blockNum, block_size>>>(
                    g.n,
                    g.d_row_offsets,
                    g.d_column_indices,
                    activated.d_elems,
                    curLevel,
                    levels.d_elems,             
                    visited.d_elems,
                    update.d_elems,
                    mask,
                    group_size,
                    group_per_block);
            }

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel_update failed ", __FILE__, __LINE__)) break;

            // spawn as many threads as the vertices in the graph
            blockNum = (g.n  % block_size == 0 ? 
                g.n / block_size :
                g.n / block_size + 1);

            // safe belt: grid width has a limit of 65535
            if (blockNum > 65535) blockNum = 65535;

            BFSKernel_update_round<<<blockNum, block_size>>>(
                g.n,
                g.d_row_offsets,
                g.d_column_indices,
                activated.d_elems,
                visited.d_elems,
                update.d_elems,     
                terminate.d_elem);

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel_expand failed ", __FILE__, __LINE__)) break;

            // timer end
            gpu_timer.stop();
            level_millis += gpu_timer.elapsedMillis();
            if (instrument) printf("[round]%d\t%f\n", i, gpu_timer.elapsedMillis());


        }

        total_milllis += level_millis;
        if (instrument) printf("%d\t%f\n", curLevel, level_millis);

        curLevel += 1;

    }
    
    printf("GPU round bitmasked bfs terminates\n");
    float billion_edges_per_second = (float)g.m / total_milllis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_milllis / 1000.0, billion_edges_per_second);


    levels.print_log();

    levels.del();
    visited.del();
    activated.del();
    update.del();
    terminate.del();
    
}


} // BFS
} // Morgen