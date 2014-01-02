
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

#include "cuda_util.cuh"
#include "util.cuh"
#include <cuda_runtime_api.h>

#define INF -1


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
void BFSGraph_gpu_queue(graph<VertexId, SizeT, Value> &g, VertexId source)
{

	// To make better use of the workset, we create two.
	// Instead of creating a new one everytime in each BFS level,
	// we just expand vertices from one to another
    queued<VertexId, SizeT> workset1(g.n);
    queued<VertexId, SizeT> workset2(g.n);


    // Initalize auxiliary list
    list<Value, SizeT> levels(g.n);
    levels.all_to(INF);


    // visitation list: 0 for unvisited
    list<int, SizeT> visited(g.n);
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
	int blockSize = 256;


	printf("gpu queued bfs starts\n");	

	while (worksetSize > 0) {

		lastWorksetSize = worksetSize;

		// spawn minimal software blocks to cover the workset
		blockNum = (worksetSize % blockSize == 0 ? 
			worksetSize / blockSize :
			worksetSize / blockSize + 1);

		// kick off timer first
		GpuTimer gpu_timer;
		gpu_timer.start();

		if (curLevel % 2 == 0) 
		{

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

			if (HandleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

			worksetSize = workset2.size();

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

			if (HandleError(cudaThreadSynchronize(), "BFSKernel failed ", __FILE__, __LINE__)) break;

		 	
		 	worksetSize = workset1.size();
		 }

		 // timer end
		 gpu_timer.stop();
		 printf("%d\t%d\t%f\n", curLevel, lastWorksetSize, gpu_timer.elapsedMillis());
		 curLevel += 1;

	}
    
    printf("gpu queued bfs terminates\n");	


    levels.print_log();

    levels.del();
    visited.del();
    workset1.del();
	workset2.del();
	
}