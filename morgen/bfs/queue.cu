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

#include <morgen/utils/timing.cuh>
#include <morgen/utils/list.cuh>
#include <morgen/workset/queue.cuh>

#include <cuda_runtime_api.h>


#define INF    -1
#define WARP_S 32

namespace morgen {

namespace bfs {

template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel(SizeT     *row_offsets,
          VertexId  *column_indices,
          VertexId  *worksetFrom,
          SizeT     *sizeFrom,
          VertexId  *worksetTo,
          SizeT     *sizeTo,
          Value     *levels,
          Value     curLevel,
          int       *visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // writing to an empty buffer
    if (tid == 0) *sizeTo = 0;
    __syncthreads();

    if (tid < *sizeFrom) {
        
        // read the who-am-I info from the workset
        VertexId outNode = worksetFrom[tid];
        levels[outNode] = curLevel;

        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];

        // serial expansion
        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            VertexId inNode = column_indices[edge];

            int old = atomicExch( (int*)&visited[inNode], 1 );

            if (old == 0) { 
                // fine-grained allocation
                SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
                worksetTo[pos] = inNode;
            }
        }   

    }
}


template<typename VertexId, typename SizeT, typename Value>
__global__ void
BFSKernel_warp_mapped(SizeT     *row_offsets,
                      VertexId  *column_indices,
                      VertexId  *worksetFrom,
                      SizeT     *sizeFrom,
                      VertexId  *worksetTo,
                      SizeT     *sizeTo,
                      Value     *levels,
                      Value     curLevel,
                      int       *visited)
{



    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    int warp_offset = tid % WARP_S;
    int warp_id     = tid / WARP_S;

    // writing to an empty buffer
    if (tid == 0) *sizeTo = 0;

    __syncthreads();


    // each warp has one variable to record how many neighbor nodes
    // it will process
    // warps per block: thread per block / warp size

    volatile __shared__ SizeT edge_first[8];
    volatile __shared__ SizeT edge_last[8];


    if (warp_id < *sizeFrom) {

        // First thread in a warp do this job
        // read out info from global mem to local mem
        if (warp_offset == 0) {
            VertexId outNode = worksetFrom[warp_id];
            levels[outNode] = curLevel;


            edge_first[warp_id % 8] = row_offsets[outNode];
            edge_last[warp_id % 8] = row_offsets[outNode+1];
        }

        __syncthreads();

        
        // in case the neighbor number > warp size
        for (SizeT edge = edge_first[warp_id % 8] + warp_offset;
             edge < edge_last[warp_id % 8];
             edge += WARP_S)
        {
            
            VertexId inNode = column_indices[edge];

            int old = atomicExch( (int*)&visited[inNode], 1 );

            if (old == 0) { 
                // fine-grained allocation
                SizeT pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
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
    int blockSize,
    bool warp_mapped)
{

    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Queue<VertexId, SizeT> workset1(g.n);
    workset::Queue<VertexId, SizeT> workset2(g.n);


    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to(INF);


    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);


    // traverse from source node
    workset1.append(source);   
    levels.set(source, 0);
    visited.set(source, 1);
    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 0;
    Value curLevel = 0;

    // kernel configuration
    int blockNum = 16;
    

    printf("gpu queued bfs starts... \n");  

    if (instrument) printf("level\tfrontier_size\tblock_num\ttime\n");


    float total_milllis = 0.0;

    // how many threads are mapped to a work element
    int mapping_factor = (warp_mapped) ? WARP_S : 1; 

    while (worksetSize > 0) {

        lastWorksetSize = worksetSize;


        // spawn minimal software blocks to cover the workset
        blockNum = (worksetSize * mapping_factor % blockSize == 0 ? 
            worksetSize * mapping_factor / blockSize :
            worksetSize * mapping_factor / blockSize + 1);


        // kick off timer first
        util::GpuTimer gpu_timer;
        gpu_timer.start();

        if (curLevel % 2 == 0) 
        {


            if (warp_mapped) {
                BFSKernel_warp_mapped<<<blockNum, blockSize>>>(g.d_row_offsets,
                                                               g.d_column_indices,
                                                               workset1.d_elems,
                                                               workset1.d_sizep,
                                                               workset2.d_elems,
                                                               workset2.d_sizep,
                                                               levels.d_elems,
                                                               curLevel,     
                                                               visited.d_elems);

            } else {
                // call kernel with device pointers
                BFSKernel<<<blockNum, blockSize>>>(g.d_row_offsets,
                                                   g.d_column_indices,
                                                   workset1.d_elems,
                                                   workset1.d_sizep,
                                                   workset2.d_elems,
                                                   workset2.d_sizep,
                                                   levels.d_elems,
                                                   curLevel,     
                                                   visited.d_elems);
            }


            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

            worksetSize = workset2.size();

         } else {

            if (warp_mapped) {
                BFSKernel_warp_mapped<<<blockNum , blockSize>>>(g.d_row_offsets,
                                                                g.d_column_indices,
                                                                workset2.d_elems,
                                                                workset2.d_sizep,
                                                                workset1.d_elems,
                                                                workset1.d_sizep,
                                                                levels.d_elems,
                                                                curLevel,     
                                                                visited.d_elems);

            } else {
                BFSKernel<<<blockNum, blockSize>>>(g.d_row_offsets,
                                                   g.d_column_indices,
                                                   workset2.d_elems,
                                                   workset2.d_sizep,
                                                   workset1.d_elems,
                                                   workset1.d_sizep,
                                                   levels.d_elems,
                                                   curLevel,
                                                   visited.d_elems);
            }



            if (util::handleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

            
            worksetSize = workset1.size();
         }

         // timer end
         gpu_timer.stop();

         if (instrument) printf("%d\t%d\t%d\t%f\n", curLevel, lastWorksetSize, blockNum, gpu_timer.elapsedMillis());
         
         total_milllis += gpu_timer.elapsedMillis();
         curLevel += 1;

    }
    
    printf("gpu queued bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_milllis / 1000000.0;
    printf("time(s): %f   speed(BE/s): %f\n", total_milllis / 1000.0, billion_edges_per_second);


    levels.print_log();

    levels.del();
    visited.del();
    workset1.del();
    workset2.del();
    
}


} // BFS

} // Morgen