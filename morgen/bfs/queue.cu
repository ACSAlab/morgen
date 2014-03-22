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
#include <morgen/utils/metrics.cuh>
#include <morgen/workset/queue.cuh>
#include <cuda_runtime_api.h>


namespace morgen {

namespace bfs {



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
BFSKernel_queue_group_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    VertexId  *worksetTo,
    SizeT     *sizeTo,
    Value     *levels,
    Value     curLevel,
    int       group_size,
    float     group_per_block)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int group_offset = tid % group_size;
    int group_id     = tid / group_size;


    // Since the workset can easily exceed 65536, we just let grouped-threads
    // iterate over a large workset
    for (int g = group_id; g < *sizeFrom; g += group_per_block * gridDim.x) {


        VertexId outNode = worksetFrom[g];
        SizeT edgeFirst = row_offsets[outNode];
        SizeT edgeLast = row_offsets[outNode+1];

        // in case the neighbor number > warp size
        for (SizeT edge = edgeFirst + group_offset; edge < edgeLast; edge += group_size)
        {
            
            VertexId inNode = column_indices[edge];

            if (levels[inNode] == MORGEN_INF) {
                levels[inNode] = curLevel + 1;
                SizeT pos = atomicAdd( (SizeT*) &(*sizeTo), 1 );
                worksetTo[pos] = inNode;
            }

        }
    }
}




template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_queue(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    VertexId source,
    bool instrument,
    int block_size,
    int group_size,
    bool get_metrics)
{

    workset::Queue<VertexId, SizeT>  workset1(g.n);
    workset::Queue<VertexId, SizeT>  workset2(g.n);



    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);


    workset1.init(source);   
    levels.set(source, 0);
    
    SizeT worksetSize = 1;
    Value curLevel = 0;


    float total_millis = 0.0;
    float expand_millis = 0.0;
    float compact_millis = 0.0;

    float group_per_block = (float) block_size / group_size;

    printf("GPU queued bfs starts... \n");  
    if (instrument) printf("level\tfrontier_size\tedge frontier\ttime\n");


    util::Metrics<VertexId, SizeT, Value> metric;
    util::Metrics<VertexId, SizeT, Value> level_metric;


    util::GpuTimer gpu_timer;
    util::GpuTimer expand_timer;
    util::GpuTimer compact_timer;


    gpu_timer.start();


    while (worksetSize > 0) {

        // 1 -> 2
        if (curLevel % 2 == 0) {

            workset2.clear_size();

            int blockNum = MORGEN_BLOCK_NUM_SAFE(worksetSize * group_size, block_size);
            BFSKernel_queue_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset1.d_elems,
                workset1.d_sizep,
                workset2.d_elems,
                workset2.d_sizep,
                levels.d_elems,
                curLevel,     
                group_size,
                group_per_block);



            worksetSize = workset2.size();
            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        } else {
        // 2 -> 1
            workset1.clear_size();


            int blockNum = MORGEN_BLOCK_NUM_SAFE(worksetSize * group_size, block_size);
            BFSKernel_queue_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset2.d_elems,
                workset2.d_sizep,
                workset1.d_elems,
                workset1.d_sizep,
                levels.d_elems,
                curLevel,     
                group_size,
                group_per_block);

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;


            worksetSize = workset1.size();



        }

//        printf("%d\n", worksetSize);


        curLevel += 1;

    } // endwhile


    gpu_timer.stop();
    total_millis = gpu_timer.elapsedMillis();

    printf("GPU queued bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);
    //printf("Accumulated Blocks: \t%d\n", accumulatedBlocks);
    if (instrument) printf("Expand:\t%f\t%f\n", expand_millis / 1000.0, compact_millis / 1000.0);
    if (instrument) metric.display();


    levels.print_log();

    levels.del();
    workset1.del();

    workset2.del();

    
}


} // BFS

} // Morgen