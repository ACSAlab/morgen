
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
template<typename VertexId, 
         typename SizeT,
         typename Value,
         bool ORDERED>
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
            Value level = curLevel + 1;

            if (ORDERED) {
                // resolve concurrent discovery through atomical operations
                int old = atomicExch( (int*)&visited[inNode], 1 );
                if (old == 0) { 
                    levels[inNode] = curLevel + 1;
                    // hash the pos by inNode id
                    int hash = outdegrees[inNode];
                    if (hash >= 0) { // ignore the 0-degree nodes
                        SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                        workset_to[slot_offsets_to[hash] + pos] = inNode;

                    }
                }
            } else {
                if (levels[inNode] > level) {
                    levels[inNode] = level;
                    int hash = outdegrees[inNode];
                    if (hash >= 0) { // ignore the 0-degree nodes
                        SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                        workset_to[slot_offsets_to[hash] + pos] = inNode;
                    }
                }
            }
        }   
    }
    
}




template<typename VertexId, 
         typename SizeT, 
         typename Value,
         bool ORDERED>
__global__ void
BFSKernel_topo_group_map(
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
  int       *outdegrees,
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


    // group_per_block * gridDim.x = how many groups of threads are spawned 
    for (int g = group_id; g < slot_sizes_from[slot_id_from]; g += group_per_block * gridDim.x) {

        // First thread in the group do this job
        //read out info from global mem to local mem
        if (group_offset == 0) {
            VertexId outNode = workset_from[slot_offsets_from[slot_id_from] + g];
            edge_first[group_id % group_per_block] = row_offsets[outNode];
            edge_last[group_id % group_per_block] = row_offsets[outNode+1];
        }

        __syncthreads();

        // serial expansion
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
                    // hash the pos by inNode id
                    int hash = outdegrees[inNode];
                    if (hash >= 0) { // ignore the 0-degree nodes(whose value = -1)
                        SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                        workset_to[slot_offsets_to[hash] + pos] = inNode;
                        //printf("I am thread %d of group %d, I am writing node %d to slot %d @ (%d, %d)",
                        //        tid, group_id, inNode, hash, slot_offsets_to[hash], pos);
                    }
                }
            } else {
                if (levels[inNode] > level) {
                    levels[inNode] = level;
                    int hash = outdegrees[inNode];
                    if (hash >= 0) { // ignore the 0-degree nodes
                        SizeT pos= atomicAdd( (SizeT*) &(slot_sizes_to[hash]), 1 );
                        workset_to[slot_offsets_to[hash] + pos] = inNode;
                    }
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
    bool unordered)
{


    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::TopoHash<VertexId, SizeT, Value>  workset[] = {
        workset::TopoHash<VertexId, SizeT, Value>(stats),
        workset::TopoHash<VertexId, SizeT, Value>(stats),
    };

    // create a outdegree table first
    // outdegree:     0  1  2-3  4-7  8-15  16-31
    // altered       -1  0  1    2    3     4
    // []
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
    int accumulatedBlocks = 0;

    // kernel configuration
    int blockNum = 16;
    printf("GPU topology-aware bfs starts... \n");  

    if (instrument) printf("level\tslot_size\tfrontier_size\tratio\ttime\n");

    float total_millis = 0.0;


    workset[0].info();


    while (worksetSize > 0) {

        lastWorksetSize = worksetSize;

        float level_millis = 0.0;

        // clear the next workset on CPU side
        for (int i = 0; i < workset[selector ^ 1].slot_num; i++) {
            workset[selector ^ 1].slot_sizes[i] = 0;
        }

        // expand edges slot by slot
        // i:           0  1  2  3  4   5   6...
        // group_size:  1  2  4  8  16  32  32
        for (int i = 0; i < workset[selector].slot_num; i++) {

            // kick off timer first
            util::GpuTimer gpu_timer;
            gpu_timer.start();

            int partialWorksetSize = workset[selector].slot_sizes[i];

            // skip the empty slot
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
                case 6: group_size = 64; break;
                case 7: group_size = 128; break;
                case 8: group_size = 256;break;
                default: group_size = 256;
            }

            // will be used in the kernel
            int group_per_block = block_size / group_size;

            // In hashed version,  the worksetSize is the logical size
            // of the hash table(smallest among the slot sizes)
            blockNum = ((partialWorksetSize * group_size) % block_size == 0 ? 
                partialWorksetSize * group_size / block_size :
                partialWorksetSize * group_size / block_size + 1);

            // safe belt: grid width has a limit of 65535
            if (blockNum > 65535) blockNum = 65535;


            if (group_size == 1) {
                BFSKernel_topo_thread_map<VertexId, SizeT, Value, true><<<blockNum, block_size>>>(
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
            } else {
                BFSKernel_topo_group_map<VertexId, SizeT, Value, true><<<blockNum, block_size>>>(
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
                    outdegrees.d_elems,
                    group_size,
                    group_per_block);

                if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;
            }
            gpu_timer.stop();
            level_millis += gpu_timer.elapsedMillis();
            accumulatedBlocks += blockNum;

            if (instrument) printf("\t[slot] %d\t%d\t%f\n", i, partialWorksetSize, gpu_timer.elapsedMillis());
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
    printf("Accumulated Blocks: \t%d\n", accumulatedBlocks);

    levels.print_log();

    levels.del();
    visited.del();
    workset[0].del();
    workset[1].del();
    
}


} // BFS
} // Morgen