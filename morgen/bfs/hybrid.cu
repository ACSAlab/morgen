
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
#include <morgen/utils/log.cuh>
#include <morgen/utils/metrics.cuh>
#include <morgen/workset/hash.cuh>
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
BFSKernel_hybrid_from_queue_group_map(
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
BFSKernel_hybrid_to_queue_gen_workset(
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




template<typename VertexId, 
         typename SizeT, 
         typename Value>
__global__ void
BFSKernel_hybrid_from_hash_group_map(
  SizeT     *row_offsets,
  VertexId  *column_indices,
  VertexId  *workset_from,
  SizeT     *slot_offsets_from,
  SizeT     *slot_sizes_from,
  int       slot_id_from,
  Value     *levels,
  Value     curLevel,
  int       *update,
  int       group_size,
  float     group_per_block)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int group_offset = tid % group_size;
    int group_id     = tid / group_size;



    // group_per_block * gridDim.x = how many groups of threads are spawned 
    for (int g = group_id; g < slot_sizes_from[slot_id_from]; g += group_per_block * gridDim.x) {

        VertexId outNode = workset_from[slot_offsets_from[slot_id_from] + g];
        SizeT edgeFirst = row_offsets[outNode];
        SizeT edgeLast = row_offsets[outNode+1];

        // serial expansion
        for (SizeT edge = edgeFirst + group_offset; edge < edgeLast; edge += group_size) 
        {

            VertexId inNode = column_indices[edge];

            if (levels[inNode] == MORGEN_INF) {
                levels[inNode] = curLevel + 1;
                update[inNode] = 1;
            }

        } // edge loop

    }
    
}



/**
 * use update[] to mask activated[]
 */
template<typename VertexId, typename SizeT>
__global__ void
BFSKernel_hybrid_to_hash_gen_workset(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *update,
    VertexId  *workset_to,
    SizeT     *slot_offsets_to,
    VertexId  *slot_sizes_to)
{
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_size) {

        if (update[tid] == 1) {

            update[tid] = 0;     // clear after activating

            SizeT outdegree = row_offsets[tid+1] - row_offsets[tid];
            int hash = log2((float)outdegree);
            if (hash >= 0) {
                SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                workset_to[slot_offsets_to[hash] + pos] = tid;
            }
        }
    }
}



template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_hybrid(
    const graph::CsrGraph<VertexId, SizeT, Value> &g, 
    VertexId source, 
    const util::Stats<VertexId, SizeT, Value> &stats,
    bool instrument,
    int block_size,
    bool get_metrics,
    int  static_group_size,
    int threshold,
    int theta,
    int alpha = 100)
{

    workset::Hash<VertexId, SizeT, Value>  workset_hash(stats, alpha);
    workset::Queue<VertexId, SizeT>  workset_queue(g.n);


    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    workset_queue.init(source); 
    levels.set(source, 0);
    util::List<int, SizeT> update(g.n);
    update.all_to(0);


    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 0;
    Value curLevel = 0;

    int blockNum;

    printf("GPU topology-aware bfs starts... \n");  

    if (instrument) printf("level\tslot_size\tfrontier_size\tratio\ttime\n");

    float total_millis = 0.0;
    //float level_millis = 0.0;
    float queue_expand_millis = 0.0;
    float queue_compact_millis = 0.0;
    float hash_expand_millis = 0.0;
    float hash_compact_millis = 0.0;

    // kernel configuration

    float group_per_block = (float)block_size / 32;

    util::Metrics<VertexId, SizeT, Value> metric;

    util::GpuTimer gpu_timer;
    util::GpuTimer queue_expand_timer;
    util::GpuTimer queue_compact_timer;
    util::GpuTimer hash_expand_timer;
    util::GpuTimer hash_compact_timer;

    cudaStream_t streams[20];
    for (int i=0; i<20; i++) {
        cudaStreamCreate(&streams[i]);
    }

    bool last_workset_is_hash = false;
    gpu_timer.start();

    while (worksetSize > 0) {

        lastWorksetSize = worksetSize;

        if (last_workset_is_hash) {

        ///////////////////////////// hash expand //////////////////////////
        if (instrument) { hash_expand_timer.start(); }

        for (int i = 0; i < workset_hash.slot_num; i++) {

            int partialWorksetSize = workset_hash.slot_sizes[i];
            if (partialWorksetSize== 0) continue;
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

            while (group_size * partialWorksetSize < threshold) {
                if (group_size == 32) break;
                group_size *= 2;
            }

            float group_per_block = (float)block_size / group_size;
            blockNum = MORGEN_BLOCK_NUM_SAFE(partialWorksetSize * group_size, block_size);
            BFSKernel_hybrid_from_hash_group_map<VertexId, SizeT, Value><<<blockNum, block_size, 0, streams[i]>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset_hash.d_elems,
                workset_hash.d_slot_offsets,
                workset_hash.d_slot_sizes,
                i,                                    
                levels.d_elems,
                curLevel,     
                update.d_elems,
                group_size,
                group_per_block);
        }

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;


        if (instrument) {
            hash_expand_timer.stop();
            hash_expand_millis +=  hash_expand_timer.elapsedMillis();
        }

        ///////////////////////////// hash expand //////////////////////////

        } else {   // last_workset_is_hash

        ///////////////////////////// queue expand //////////////////////////
        if (instrument) { queue_expand_timer.start(); }

        int blockNum = MORGEN_BLOCK_NUM_SAFE(worksetSize * 32, block_size);
        BFSKernel_hybrid_from_queue_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
            g.d_row_offsets,
            g.d_column_indices,
            workset_queue.d_elems,
            workset_queue.d_sizep,
            levels.d_elems,
            curLevel,     
            32,
            group_per_block,
            update.d_elems);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        if (instrument) {
            queue_expand_timer.stop();
            queue_expand_millis +=  queue_expand_timer.elapsedMillis();
            //printf("queue: %d\t%d\n", curLevel, lastWorksetSize);
        }

        ///////////////////////////// queue expand //////////////////////////
        
        } // till here, the next frontier is done, now we decide what is the representation

        ///////////////////////////// hash compaction //////////////////////////
        if (lastWorksetSize > theta) {

        if (instrument) { hash_compact_timer.start(); }

        workset_hash.clear_slot_sizes();

        blockNum = MORGEN_BLOCK_NUM_SAFE(g.n, block_size);
        BFSKernel_hybrid_to_hash_gen_workset<<<blockNum, block_size>>> (
            g.n,
            g.d_row_offsets,
            g.d_column_indices,
            update.d_elems,
            workset_hash.d_elems,
            workset_hash.d_slot_offsets,
            workset_hash.d_slot_sizes);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        last_workset_is_hash = true;

        worksetSize = workset_hash.sum_slot_size();

        if (instrument) {
            hash_compact_timer.stop();
            hash_compact_millis += hash_compact_timer.elapsedMillis();
            printf("hash: %d\t%d\n", curLevel, lastWorksetSize);
        }

        ///////////////////////////// hash compaction //////////////////////////

        } else {   // if (lastWorksetSize <= theta)

        ////////////////////////////////// queue compaction ///////////////////////////
        if (instrument) { queue_compact_timer.start(); }

        workset_queue.clear_size();

        blockNum = MORGEN_BLOCK_NUM_SAFE(g.n, block_size);
        BFSKernel_hybrid_to_queue_gen_workset<<<blockNum, block_size>>> (
            g.n,
            g.d_row_offsets,
            g.d_column_indices,
            update.d_elems,
            workset_queue.d_elems,
            workset_queue.d_sizep);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        worksetSize = workset_queue.size();
        last_workset_is_hash = false;
        if (instrument) {
            queue_compact_timer.stop();
            queue_compact_millis +=  queue_compact_timer.elapsedMillis();
            printf("queue: %d\t%d\n", curLevel, lastWorksetSize);
        }
        ////////////////////////////////// queue compaction ///////////////////////////

        }
        curLevel += 1;
    }

    gpu_timer.stop();
    total_millis = gpu_timer.elapsedMillis();


    printf("GPU topo bfs terminates\n");
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);


    if (instrument)
        printf("Expand: \t%f\t%f\t%f\t%f\n",
         queue_expand_millis / 1000.0,
         queue_compact_millis / 1000.0,
         hash_expand_millis / 1000.0,
         hash_compact_millis / 1000.0);

    if (get_metrics) metric.display();


    levels.print_log();
    levels.del();
    update.del();
    workset_queue.del();
    workset_hash.del();
    
}


} // BFS
} // Morgen