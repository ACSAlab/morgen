
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
#include <morgen/workset/topo_hash.cuh>

#include <cuda_runtime_api.h>


namespace morgen {

namespace bfs {


/**
 * This is a fixed thread-mapping kernel for hashe-based workset
 * The workset of current level is processed in one kernal launch
 */
template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_topo_thread_map(
  SizeT     *row_offsets,
  VertexId  *column_indices,
  VertexId  *workset_from,
  SizeT     *slot_offsets_from,
  SizeT     *slot_sizes_from,
  int       slot_id_from,
  VertexId  *workset_to,
  SizeT     *slot_offsets_to,
  VertexId  *slot_sizes_to,
  Value     *levels,
  Value     curLevel,
  int       *visited,
  int       *outdegrees)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < slot_sizes_from[slot_id_from]) {

        VertexId outNode = workset_from[slot_offsets_from[slot_id_from] + tid];
        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];

            // serial expansion
        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            VertexId inNode = column_indices[edge];

                // resolve concurrent discovery through atomical operations
            int old = atomicExch( (int*)&visited[inNode], 1 );

            if (old == 0) { 
                levels[inNode] = curLevel + 1;
                // hash the pos by inNode id
                int hash = outdegrees[inNode];
                // exclusively get the writing position within the slot
                SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                workset_to[slot_offsets_to[hash] + pos] = inNode;
            }
        }   
    }
    
}




template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_gpu_topo(
    const graph::CsrGraph<VertexId, SizeT, Value> &g, 
    VertexId source, 
    const util::Stats<VertexId, SizeT, Value> &stats,
    bool instrument = false)
{


    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another

    workset::TopoHash<VertexId, SizeT, Value>  workset[] = {
        workset::TopoHash<VertexId, SizeT, Value>(stats),
        workset::TopoHash<VertexId, SizeT, Value>(stats),
    };

    // create a outdegree table first
    util::List<Value, SizeT> outdegrees(g.n);
    for (SizeT i = 0; i < g.n; i++) {
        SizeT outDegree = g.row_offsets[i+1] - g.row_offsets[i];
        int times = 0;
        while (outDegree > 0) {
            outDegree /= 2;
            times++;
        }
        outdegrees.elems[i] = times - 1;
    }



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
    workset[0].insert(source);   
    levels.set(source, 0);
    visited.set(source, 1);

    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 0;
    Value curLevel = 0;

    // kernel configuration
    int blockNum = 16;
    int blockSize = 256;

    printf("GPU topology-aware bfs starts... \n");  
    if (instrument) printf("level\tslot_size\tfrontier_size\tratio\ttime\n");

    float total_millis = 0.0;

    while (worksetSize > 0) {

        lastWorksetSize = worksetSize;

        float level_millis = 0.0;

        // clear the next workset on CPU side
        for (int i = 0; i < workset[selector ^ 1].slot_num; i++) {
            workset[selector ^ 1].slot_sizes[i] = 0;
        }

        // expand edges slot by slot
        for (int i = 0; i < workset[selector].slot_num; i++) {

            // kick off timer first
            util::GpuTimer gpu_timer;
            gpu_timer.start();

            int partialWorksetSize = workset[selector].slot_sizes[i];

            // skip the empty slot
            if (partialWorksetSize== 0) continue;

            // In hashed version,  the worksetSize is the logical size
            // of the hash table(smallest among the slot sizes)
            blockNum = (partialWorksetSize % blockSize == 0 ? 
                partialWorksetSize / blockSize :
                partialWorksetSize / blockSize + 1);

            BFSKernel_topo_thread_map<<<blockNum, blockSize>>>(
                g.d_row_offsets,
                g.d_column_indices,
                workset[selector].d_elems,
                workset[selector].d_slot_offsets,
                workset[selector].d_slot_sizes,
                i,                                    
                workset[selector ^ 1].d_elems,
                workset[selector ^ 1].d_slot_offsets,
                workset[selector ^ 1].d_slot_sizes,
                levels.d_elems,
                curLevel,     
                visited.d_elems,
                outdegrees.d_elems);

            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

            gpu_timer.stop();
            level_millis += gpu_timer.elapsedMillis();
            if (instrument) printf("[slot] %d\t%d\t%f\n", i, partialWorksetSize, gpu_timer.elapsedMillis());
        }


        worksetSize = workset[selector ^ 1].total_size();

        // timer end
        total_millis += level_millis;
        if (instrument) printf("%d\t%d\t%f\n", curLevel, lastWorksetSize, level_millis);

        curLevel += 1;
        selector = selector ^ 1;
    }
    
    printf("GPU hashed bfs terminates\n");
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);

    levels.print_log();

    levels.del();
    visited.del();
    workset[0].del();
    workset[1].del();
    
}


} // BFS
} // Morgen