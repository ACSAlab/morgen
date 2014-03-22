
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
  VertexId  *workset_to,
  SizeT     *slot_offsets_to,
  SizeT     *slot_sizes_to,
  int       slot_id_from,
  Value     *levels,
  Value     curLevel,
  int       *update,
  int       group_size,
  float     group_per_block,
  int       *outdegrees)
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
                int hash = outdegrees[inNode];
                if (hash >= 0) {
                    SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                    workset_to[slot_offsets_to[hash] + pos] = inNode;
                }
            }
        } // edge loop

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
    int threshold=0,
    int alpha = 100)
{


    workset::Hash<VertexId, SizeT, Value>  workset1(stats, alpha);
    workset::Hash<VertexId, SizeT, Value>  workset2(stats, alpha);



    // create a outdegree table first     
    util::List<Value, SizeT> outdegreesLog(g.n);
    for (SizeT i = 0; i < g.n; i++) {
        SizeT outDegree = g.row_offsets[i+1] - g.row_offsets[i];

        int slot_should_go = util::getLogOf(outDegree);
        while (workset1.slot_size_max[slot_should_go] == 0) { 
            slot_should_go += 1;
        }
        outdegreesLog.elems[i] = slot_should_go;
    }
    outdegreesLog.transfer();

    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    // traverse from source node
    workset1.insert(outdegreesLog.elems[source], source);   
    levels.set(source, 0);
    util::List<int, SizeT> update(g.n);
    update.all_to(0);


    SizeT worksetSize = 1;
    Value curLevel = 0;
    SizeT edge_frontier_size;

    // kernel configuration
    int blockNum;
    int group_size;
    printf("GPU topology-aware bfs starts... \n");  

    if (instrument) printf("level\tslot_size\tfrontier_size\tratio\ttime\n");

    float total_millis = 0.0;
    //float level_millis = 0.0;
    float expand_millis = 0.0;
    float compact_millis = 0.0;

    util::Metrics<VertexId, SizeT, Value> metric;

    // kick off timer first
    util::GpuTimer gpu_timer;
    util::GpuTimer expand_timer;
    util::GpuTimer compact_timer;

    cudaStream_t streams[20];

    for (int i=0; i<20; i++) {
        cudaStreamCreate(&streams[i]);
    }

    gpu_timer.start();

    while (worksetSize > 0) {


        if (curLevel % 2 == 0) {

        // 1->2
        workset2.clear_slot_sizes();
        for (int i = 0; i < workset1.slot_num; i++) {

            int partialWorksetSize = workset1.slot_sizes[i];
            if (partialWorksetSize== 0) continue;

            // decide which mapping strategy to be used according to i
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

            float group_per_block = (float) block_size / group_size;
            blockNum = MORGEN_BLOCK_NUM_SAFE(partialWorksetSize * group_size, block_size);

            BFSKernel_topo_group_map<VertexId, SizeT, Value><<<blockNum, block_size, 0, streams[i]>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset1.d_elems,
                workset1.d_slot_offsets,
                workset1.d_slot_sizes,
                workset2.d_elems,
                workset2.d_slot_offsets,
                workset2.d_slot_sizes,
                i,                                    
                levels.d_elems,
                curLevel,     
                update.d_elems,
                group_size,
                group_per_block,
                outdegreesLog.d_elems);

            if (instrument) printf("\t[slot] %d\t%d\t%d\t%d\t%f\t%f\n", i, group_size, partialWorksetSize, edge_frontier_size, expand_timer.elapsedMillis());
        } // for

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        worksetSize = workset2.sum_slot_size();

        } else {

        // 2->1
        workset1.clear_slot_sizes();
        for (int i = 0; i < workset2.slot_num; i++) {

            int partialWorksetSize = workset2.slot_sizes[i];
            if (partialWorksetSize== 0) continue;

            // decide which mapping strategy to be used according to i
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

            float group_per_block = (float) block_size / group_size;
            blockNum = MORGEN_BLOCK_NUM_SAFE(partialWorksetSize * group_size, block_size);

            BFSKernel_topo_group_map<VertexId, SizeT, Value><<<blockNum, block_size, 0, streams[i]>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset2.d_elems,
                workset2.d_slot_offsets,
                workset2.d_slot_sizes,
                workset1.d_elems,
                workset1.d_slot_offsets,
                workset1.d_slot_sizes,
                i,                                    
                levels.d_elems,
                curLevel,     
                update.d_elems,
                group_size,
                group_per_block,
                outdegreesLog.d_elems);

            if (instrument) printf("\t[slot] %d\t%d\t%d\t%d\t%f\t%f\n", i, group_size, partialWorksetSize, edge_frontier_size, expand_timer.elapsedMillis());
        } // for

        if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

        worksetSize = workset1.sum_slot_size();

        }
        curLevel += 1;
    }

    gpu_timer.stop();
    total_millis = gpu_timer.elapsedMillis();

    printf("GPU topo bfs terminates\n");
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);
    //printf("Accumulated Blocks: \t%d\n", accumulatedBlocks);

    if (instrument) printf("Expand: \t%f\t%f\n", expand_millis / 1000.0, compact_millis / 1000.0);
    if (instrument) metric.display();

    levels.print_log();
    levels.del();
    update.del();
    outdegreesLog.del();
    workset1.del();
    workset2.del();

    
}


} // BFS
} // Morgen