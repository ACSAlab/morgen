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


/* texture memory */
//texture<int> tex_row_offsets;
//texture<int> tex_column_indices;


template<typename VertexId,
         typename SizeT, 
         typename Value>
__global__ void
BFSKernel_queue_thread_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    Value     *levels,
    Value     curLevel,
    int       *update)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *sizeFrom) {
        
        // read the who-am-I info from the workset
        VertexId outNode = worksetFrom[tid];

        SizeT outEdgeFirst = row_offsets[outNode];
        //SizeT outEdgeFirst = tex1Dfetch(tex_row_offsets, outNode);

        SizeT outEdgeLast = row_offsets[outNode+1];
        //SizeT outEdgeLast = tex1Dfetch(tex_row_offsets, outNode+1);

        // serial expansion
        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            VertexId inNode = column_indices[edge];
            //VertexId inNode = tex1Dfetch(tex_column_indices, edge);



            if (levels[inNode] == MORGEN_INF) {
                levels[inNode] = curLevel + 1;
                update[inNode] = 1;
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
BFSKernel_queue_group_map(
    SizeT     *row_offsets,
    VertexId  *column_indices,
    VertexId  *worksetFrom,
    SizeT     *sizeFrom,
    Value     *levels,
    Value     curLevel,
    int       group_size,
    float     group_per_block,
    int       *update)
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
            //VertexId inNode = tex1Dfetch(tex_column_indices, edge);


            if (levels[inNode] == MORGEN_INF) {
                levels[inNode] = curLevel + 1;
                update[inNode] = 1;
            }

        }
    }
}


/**
 * use update[] to mask activated[]
 */
template<typename VertexId, typename SizeT>
__global__ void
BFSKernel_queue_gen_workset(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *update,
    VertexId  *worksetTo,
    SizeT     *sizeTo)
{
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_size) {

        if (update[tid] == 1) {

            update[tid] = 0;     // clear after activating

            SizeT pos = atomicAdd( (SizeT*) &(*sizeTo), 1 );
            worksetTo[pos] = tid;
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
    bool get_metrics)
{

    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Queue<VertexId, SizeT>  workset(g.n);

    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);


    util::List<int, SizeT> update(g.n);
    update.all_to(0);

    // traverse from source node
    workset.init(source);   
    levels.set(source, 0);
    
    SizeT worksetSize = 1;
    //SizeT lastWorksetSize = 0;
    Value curLevel = 0;

    SizeT edge_frontier_size;

    float total_millis = 0.0;
    float expand_millis = 0.0;
    float compact_millis = 0.0;

    if (warp_mapped == false) group_size = 1;
    float group_per_block = (float)block_size / group_size;

    printf("GPU queued bfs starts... \n");  
    if (instrument) printf("level\tfrontier_size\tblock_num\ttime\n");


    util::Metrics<VertexId, SizeT, Value> metric;
    util::Metrics<VertexId, SizeT, Value> level_metric;

    /* 

    bind the graph in texture memory(1D)
    
    if (util::handleError(cudaBindTexture(0, tex_column_indices, g.d_column_indices, sizeof(VertexId) * g.m), 
        "CsrGraph: bindTexture(d_column_indices) failed", __FILE__, __LINE__)) exit(1);        

    if (util::handleError(cudaBindTexture(0, tex_row_offsets, g.d_row_offsets, sizeof(SizeT) * (g.n + 1)), 
        "CsrGraph: bindTexture(d_row_offsets) failed", __FILE__, __LINE__)) exit(1);
        
    printf("Done texture memory binding.\n");

    */


    util::GpuTimer gpu_timer;
    util::GpuTimer expand_timer;
    util::GpuTimer compact_timer;


    gpu_timer.start();


    while (worksetSize > 0) {

        if (instrument) {
            workset.transfer_back();
            metric.count(workset.elems, workset.size(), g, group_size);
            edge_frontier_size = 0;
            for (int i = 0; i < *workset.sizep; i++) {
                VertexId v = workset.elems[i];
                SizeT start = g.row_offsets[v];
                SizeT end = g.row_offsets[v+1];
                edge_frontier_size += (end - start);
            }
            expand_timer.start();  // start timer
        }


        // spawn minimal(but enough) software blocks to cover the workset
        int blockNum = (worksetSize * group_size % block_size == 0 ? 
            worksetSize * group_size / block_size :
            worksetSize * group_size/ block_size + 1);
        
        // safe belt: grid width has a limit of 65535
        if (blockNum > 65535) blockNum = 65535;

        if (warp_mapped) {

            BFSKernel_queue_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset.d_elems,
                workset.d_sizep,
                levels.d_elems,
                curLevel,     
                group_size,
                group_per_block,
                update.d_elems);

        } else { // thread map

            BFSKernel_queue_thread_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                g.d_row_offsets,                                        
                g.d_column_indices,
                workset.d_elems,
                workset.d_sizep,
                levels.d_elems,
                curLevel,     
                update.d_elems);

        }
        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        if (instrument) {
            expand_timer.stop();
            expand_millis += expand_timer.elapsedMillis();
            compact_timer.start();
        }      

        workset.clear_size();


        blockNum = (g.n % block_size == 0) ? 
            (g.n / block_size) :
            (g.n / block_size + 1);
        if (blockNum > 65535) blockNum = 65535;

  
        // generate the next workset according to update[]
        BFSKernel_queue_gen_workset<<<blockNum, block_size>>> (
            g.n,
            g.d_row_offsets,
            g.d_column_indices,
            update.d_elems,
            workset.d_elems,
            workset.d_sizep);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;


        if (instrument) {
            compact_timer.stop();
            compact_millis += compact_timer.elapsedMillis();
            printf("%d\t%d\t%d\t%f\t%f\n", curLevel, worksetSize, edge_frontier_size, expand_timer.elapsedMillis(), compact_timer.elapsedMillis());
        }

        worksetSize = workset.size();

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
    update.del();
    workset.del();
    
}


} // BFS

} // Morgen