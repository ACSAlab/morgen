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


#include <cuda_runtime_api.h>




namespace morgen {

namespace bfs {

/**
 * each thread wakeup and check if activated[tid] == 1
 * using update[] to mark unvisited vertices in this round
 */
template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_expand_single(
  SizeT     max_size,
  SizeT     *row_offsets,
  VertexId  *column_indices,
  int       *activated,
  Value     *levels,
  Value     curLevel,
  int       *visited,
  int       *update)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_size) {

        if (activated[tid] == 1) {

            activated[tid] = 0;     // wakeup only once
            SizeT outEdgeFirst = row_offsets[tid];
            SizeT outEdgeLast = row_offsets[tid+1];

            // serial expansion
            for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

                VertexId inNode = column_indices[edge];
                if (visited[inNode] == 0) {
                    levels[inNode] = curLevel + 1;
                    update[inNode] = 1;
                }
            }
        }
    }
}



template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_expand_group(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *activated,
    Value     *levels,
    Value     curLevel,
    int       *visited,
    int       *update,
    int       group_size,
    float     group_per_block)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int group_offset = tid % group_size;
    int group_id     = tid / group_size;

    for (int g = group_id; g < max_size; g += group_per_block * gridDim.x) {


        if (activated[g] == 1) {

            activated[g] = 0;     // wakeup only once
            SizeT outEdgeFirst = row_offsets[g];
            SizeT outEdgeLast = row_offsets[g+1];

            // serial expansion
            for (SizeT edge = outEdgeFirst + group_offset; edge < outEdgeLast; edge += group_size) {

                VertexId inNode = column_indices[edge];

                if (visited[inNode] == 0) {
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
BFSKernel_update(SizeT     max_size,
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
void BFSGraph_gpu_bitmask(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    bool instrument,
    int block_size,
    bool warp_mapped,
    int group_size)
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


    // spawn as many threads as the vertices in the graph
    int mapping_factor = (warp_mapped) ? group_size : 1; 
    float group_per_block = (float)block_size / group_size;

    int blockNum = (g.n * mapping_factor % block_size == 0 ? 
        g.n * mapping_factor / block_size :
        g.n * mapping_factor / block_size + 1);

    // safe belt: grid width has a limit of 65535
    if (blockNum > 65535) blockNum = 65535;

    printf("GPU bitmasked bfs starts... \n");   
    if (instrument) printf("level\ttime\n");

    float total_milllis = 0.0;

    // loop as long as the flag is set
    while (terminate.getVal() == 0) {

        // set true at first, if no vertex has been expanded
        // the while loop will be terminated
        terminate.set(1);

        // kick off timer first
        util::GpuTimer gpu_timer;
        gpu_timer.start();

        if (group_size == 1) {
            BFSKernel_expand_single<<<blockNum, block_size>>>(
                g.n,
                g.d_row_offsets,
                g.d_column_indices,
                activated.d_elems,
                levels.d_elems,
                curLevel,             
                visited.d_elems,
                update.d_elems);
        } else {
            BFSKernel_expand_group<<<blockNum, block_size>>>(
                g.n,
                g.d_row_offsets,
                g.d_column_indices,
                activated.d_elems,
                levels.d_elems,
                curLevel,             
                visited.d_elems,
                update.d_elems,
                group_size,
                group_per_block);
        }

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel_expand failed ", __FILE__, __LINE__)) break;

        BFSKernel_update<<<blockNum, block_size>>>(
            g.n,
            g.d_row_offsets,
            g.d_column_indices,
            activated.d_elems,
            visited.d_elems,
            update.d_elems,     
            terminate.d_elem);
        
        if (util::handleError(cudaThreadSynchronize(), "BFSKernel_update failed ", __FILE__, __LINE__)) break;


         // timer end
         gpu_timer.stop();

         if (instrument) printf("%d\t%f\n", curLevel, gpu_timer.elapsedMillis());
         total_milllis += gpu_timer.elapsedMillis();
         curLevel += 1;

    }
    
    printf("GPU bitmasked bfs terminates\n");
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