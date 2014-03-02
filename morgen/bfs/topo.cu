
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
 * This is a fixed thread-mapping kernel for hashe-based workset
 * The workset of current level is processed in one kernal launch
 */
template<typename VertexId, 
         typename SizeT,
         typename Value>
__global__ void
BFSKernel_topo_thread_map(
  SizeT     *row_offsets,
  VertexId  *column_indices,
  VertexId  *workset_from,
  SizeT     *slot_offsets_from,
  SizeT     *slot_sizes_from,
  int       slot_id_from,
  Value     *levels,
  Value     curLevel,
  int       *visited,
  int       *update)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < slot_sizes_from[slot_id_from]) {

        VertexId outNode = workset_from[slot_offsets_from[slot_id_from] + tid];
        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];


        for (SizeT e = outEdgeFirst; e < outEdgeLast; e++) {
            VertexId inNode = column_indices[e];

            if (levels[inNode] == MORGEN_INF) { 
                levels[inNode] = curLevel + 1;
                update[inNode] = 1;
            }
            

       }
    }   
    
    
}




template<typename VertexId, 
         typename SizeT, 
         typename Value>
__global__ void
BFSKernel_topo_group_map(
  SizeT     *row_offsets,
  VertexId  *column_indices,
  VertexId  *workset_from,
  SizeT     *slot_offsets_from,
  SizeT     *slot_sizes_from,
  int       slot_id_from,
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
BFSKernel_topo_gen_workset(
    SizeT     max_size,
    SizeT     *row_offsets,
    VertexId  *column_indices,
    int       *visited,
    int       *update,
    VertexId  *workset_to,
    SizeT     *slot_offsets_to,
    VertexId  *slot_sizes_to,
    int       *outdegrees)
{
    int tid =  blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < max_size) {

        if (update[tid] == 1) {

            update[tid] = 0;     // clear after activating
            visited[tid] = 1;

            int hash = outdegrees[tid];
            if (hash >= 0) {
                SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                workset_to[slot_offsets_to[hash] + pos] = tid;
            }
        }
    }
}





template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_topo(
    const graph::CsrGraph<VertexId, SizeT, Value> &g, 
    VertexId source, 
    const util::Stats<VertexId, SizeT, Value> &stats,
    bool instrument,
    int block_size,
    bool get_metrics,
    int  static_group_size,
    int threshold)
{


    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Hash<VertexId, SizeT, Value>  workset(stats);


    // create a outdegree table first
    // outdegree:     0  (0,1]  (1, 2]  (2, 4]   (4, 8]   (8, 16]
    // altered       -1   0      1       2       3        4       
    util::List<Value, SizeT> outdegreesLog(g.n);
    for (SizeT i = 0; i < g.n; i++) {
        SizeT outDegree = g.row_offsets[i+1] - g.row_offsets[i];

        
        
        if (outDegree == 0) 
            outdegreesLog.elems[i] = -1;
        else if (outDegree > 0 && outDegree <= 1)
            outdegreesLog.elems[i] = 0;
        else if (outDegree > 1 && outDegree <= 2)
            outdegreesLog.elems[i] = 1;
        else if (outDegree > 2 && outDegree <= 4)
            outdegreesLog.elems[i] = 2;
        else if (outDegree > 4 && outDegree <= 8)
            outdegreesLog.elems[i] = 3;
        else if (outDegree > 8 && outDegree <= 16)
            outdegreesLog.elems[i] = 4;
        else if (outDegree > 16 && outDegree <= 32)
            outdegreesLog.elems[i] = 5;
        else 
            outdegreesLog.elems[i] = 5;
        
        //outdegreesLog.elems[i] = util::getLogOf(outDegree);;
    }
    outdegreesLog.transfer();



    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);


    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);

    // traverse from source node
    workset.insert(outdegreesLog.elems[source], source);   
    levels.set(source, 0);
    visited.set(source, 1);
    util::List<int, SizeT> update(g.n);
    update.all_to(0);


    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 0;
    Value curLevel = 0;
    //int accumulatedBlocks = 0;

    // kernel configuration
    int blockNum;
    printf("GPU topology-aware bfs starts... \n");  

    if (instrument) printf("level\tslot_size\tfrontier_size\tratio\ttime\n");

    float total_millis = 0.0;
    float level_millis = 0.0;
    float expand_millis = 0.0;
    float compact_millis = 0.0;

    util::Metrics<VertexId, SizeT, Value> metric;

    // kick off timer first
    util::GpuTimer gpu_timer;
    util::GpuTimer expand_timer;
    util::GpuTimer compact_timer;


    gpu_timer.start();

    while (worksetSize > 0) {

        lastWorksetSize = worksetSize;

        if (instrument) level_millis = 0;

        //float level_millis = 0.0;

        // expand edges slot by slot
        // i:           0  1  2  3  4   5   6...
        // group_size:  1  2  4  8  16  32  32
        for (int i = 0; i < workset.slot_num; i++) {

            // kick off timer first
            if (instrument) {
                expand_timer.start();
            }

            int partialWorksetSize = workset.slot_sizes[i];
            if (partialWorksetSize== 0) continue;


            // decide which mapping strategy to be used according to i
            int group_size = 0;
            switch (i) {
                case 0: group_size = 1; break;
                case 1: group_size = 2; break;
                case 2: group_size = 4; break;
                case 3: group_size = 8; break;
                case 4: group_size = 16; break;
                case 5: group_size = 32; break;
                default: group_size = 32; 
                /*
                case 5: group_size = 32; break;
                case 6: group_size = 64; break;
                case 7: group_size = 128; break;
                case 8: group_size = 256;break;
                case 9: group_size = 512; break;
                case 10: group_size = 1024;break;
                case 11: group_size = 2048; break;
                case 12: group_size = 4096; break;
                case 13: group_size = 8192; break;
                case 14: group_size = 16384; break;
                case 15: group_size = 32768; break;
                default: fprintf(stderr, "out of control!!\n"); return;*/
            }

            while (group_size * partialWorksetSize < threshold) {
                if (group_size == 32) break;
                group_size *= 2;
            }

            if (static_group_size != 0) group_size = static_group_size;

            if (get_metrics) {
                workset.transfer_back();
                metric.count(workset.elems + workset.slot_offsets[i], partialWorksetSize, g, group_size);
            }


            float group_per_block = (float)block_size / group_size;
            blockNum = ((partialWorksetSize * group_size) % block_size == 0 ? 
                partialWorksetSize * group_size / block_size :
                partialWorksetSize * group_size / block_size + 1);
            // safe belt
            if (blockNum > 65535) blockNum = 65535;


            if (group_size == 1) {
                BFSKernel_topo_thread_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                    g.d_row_offsets,
                    g.d_column_indices,
                    workset.d_elems,
                    workset.d_slot_offsets,
                    workset.d_slot_sizes,
                    i,                                    
                    levels.d_elems,
                    curLevel,     
                    visited.d_elems,
                    update.d_elems);
            } else {
                BFSKernel_topo_group_map<VertexId, SizeT, Value><<<blockNum, block_size>>>(
                    g.d_row_offsets,
                    g.d_column_indices,
                    workset.d_elems,
                    workset.d_slot_offsets,
                    workset.d_slot_sizes,
                    i,                                    
                    levels.d_elems,
                    curLevel,     
                    visited.d_elems,
                    update.d_elems,
                    group_size,
                    group_per_block);
            }

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

            if (instrument) { 
                expand_timer.stop(); 
                expand_millis +=  expand_timer.elapsedMillis();
                level_millis += expand_timer.elapsedMillis();
            }
            //level_millis += gpu_timer.elapsedMillis();
            //if (instrument) printf("\t[slot] %d\t%d\t%d\t%f\n", i, group_size, partialWorksetSize, expand_timer.elapsedMillis());
        }


        if (instrument) compact_timer.start();

        // clear the workset first
        workset.clear_slot_sizes();

        blockNum = (g.n % block_size == 0) ? 
            (g.n / block_size) :
            (g.n / block_size + 1);
        if (blockNum > 65535) blockNum = 65535;

        // generate the next workset according to update[]
        BFSKernel_topo_gen_workset<<<blockNum, block_size>>> (
            g.n,
            g.d_row_offsets,
            g.d_column_indices,
            visited.d_elems,
            update.d_elems,
            workset.d_elems,
            workset.d_slot_offsets,
            workset.d_slot_sizes,
            outdegreesLog.d_elems);

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        // get the size of workset
        worksetSize = workset.sum_slot_size();

        if (instrument) {
            compact_timer.stop();
            compact_millis +=  compact_timer.elapsedMillis();
            printf("%d\t%d\t%f\t%f\n", curLevel, lastWorksetSize, level_millis, compact_timer.elapsedMillis());
        }
        //total_millis += level_millis;
        curLevel += 1;
    }

    gpu_timer.stop();
    total_millis = gpu_timer.elapsedMillis();


    printf("GPU topo bfs terminates\n");
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);
    //printf("Accumulated Blocks: \t%d\n", accumulatedBlocks);


    if (instrument) printf("Expand: %f\t%f\n", expand_millis / 1000.0, compact_millis / 1000.0);

    if (get_metrics) metric.display();


    levels.print_log();

    levels.del();
    visited.del();
    update.del();
    outdegreesLog.del();
    workset.del();
    workset.del();
    
}


} // BFS
} // Morgen