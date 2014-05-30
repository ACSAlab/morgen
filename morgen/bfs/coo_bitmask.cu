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


#include <morgen/graph/coo_edge_tuple.cuh>


#include <cuda_runtime_api.h>




namespace morgen {

namespace bfs {

/**
 * each thread wakeup and check if activated[tid] == 1
 * using update[] to mark unvisited vertices in this round
 */
template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_expand_coo(
  SizeT            max_size,
  morgen::graph::CooEdgeTuple<VertexId>     *elems,
  int              *activated,
  Value            *levels,
  Value            curLevel,
  int              *visited,
  int              *update)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_size) {

        VertexId src = elems[tid].row;

        if (activated[src] == 1) {
            activated[src] = 0;     // wakeup only once
            VertexId dest = elems[tid].col;
            if (visited[dest] == 0) {
                levels[dest] = curLevel + 1;
                update[dest] = 1;
            }
        }
    }
}



/**
 * use update[] to mask activated[]
 */
template<typename SizeT>
__global__ void
BFSKernel_update_coo(
    SizeT     max_size,
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
            *terminate = 0; 
        }
    }
}


template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_bitmask_coo(
    const graph::CooGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    int block_size,
    int instrument)
{

    // use a list to represent bitmask
    util::List<int, SizeT> activated(g.n);
    util::List<int, SizeT> update(g.n);
    util::List<Value, SizeT> levels(g.n);
    util::List<int, SizeT> visited(g.n);
    util::Var<int> terminate;

    activated.all_to(0);
    update.all_to(0);
    levels.all_to((Value) MORGEN_INF);
    visited.all_to(0);
    terminate.set(0);

    // traverse from source node
    activated.set(source, 1);
    levels.set(source, 0);
    visited.set(source, 1);
    Value curLevel = 0;



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


        int blockNum = MORGEN_BLOCK_NUM_SAFE(g.m, block_size);
        BFSKernel_expand_coo<<<blockNum, block_size>>>(
            g.m,
            g.d_elems,
            activated.d_elems,
            levels.d_elems,
            curLevel,             
            visited.d_elems,
            update.d_elems);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel_expand failed ", __FILE__, __LINE__)) break;



        blockNum = MORGEN_BLOCK_NUM_SAFE(g.n, block_size);
        BFSKernel_update_coo<<<blockNum, block_size>>>(
            g.n,
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