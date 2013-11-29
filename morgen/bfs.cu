
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

	// write to an empty workset
	*sizeTo = 0;

	for (int j = tid; j < *sizeFrom; j += blockDim.x * gridDim.x) {

		// read the who-am-I info from the workset, visit it immediately
		VertexId outNode = worksetFrom[j];

		levels[outNode] = curLevel;

		SizeT outEdgeStart = row_offsets[j];
		SizeT outEdgeEnd = row_offsets[j+1];

		// In this implemention, parallelism is exploited 
		// at a coarse granularity size
		for (int i = outEdgeStart; i < outEdgeEnd; i++) {
			VertexId inNode = column_indices[i];

			int old = atomicExch( (int*)&visited[inNode], 1 );

			if (old == 0) {	
				int pos= atomicAdd( (SizeT*) &(*sizeTo), 1 );
				worksetTo[pos] = inNode;
			}
		}		
	}
}


template<typename VertexId, typename SizeT, typename Value>
void BFSGraph(graph<VertexId, SizeT, Value> &g, VertexId start)
{

	// To make better use of the workset, we create two.
	// Instead of creating a new one everytime in each BFS level,
	// we just expand vertices from one to another


    subvertices<VertexId, SizeT> workset1(g.n);
    subvertices<VertexId, SizeT> workset2(g.n);


    // Initalize auxiliary list
    list<Value, SizeT> levels(g.n);
    levels.all_to(INF);


    list<int, SizeT> visited(g.n);
    visited.all_to(0);


	// from source node
    workset1.append(start);   
    visited.set(start, 1);



    SizeT size = 1;
	Value curLevel = 0;
	// kernel configuration
	int blockNum = 16;
	int blockSize = 256;


	while (size != 0) {

		if (curLevel % 2 == 0) 
		{
			BFSKernel<<<blockNum, blockSize>>>(g.d_row_offsets,
				                               g.d_column_indices,
				                               workset1.d_vertices,
				                               workset1.d_sizep,
				                               workset2.d_vertices,
				                               workset2.d_sizep,
				                               levels.d_elements,
				                               curLevel,     
				                               visited.d_elements);

			if (HandleError(cudaThreadSynchronize(), "BFSKernel failed ",
			                __FILE__, __LINE__)) break;

			workset2.print();              
			size = *workset2.sizep;
			printf("next size: %d\n", size);
		 } else {
		 	BFSKernel<<<blockNum, blockSize>>>(g.d_row_offsets,
		 		                               g.d_column_indices,
		 		                               workset2.d_vertices,
		 		                               workset2.d_sizep,
		 		                               workset1.d_vertices,
		 		                               workset1.d_sizep,
		 		                               levels.d_elements,
		 		                               curLevel,
		 		                               visited.d_elements);

			if (HandleError(cudaThreadSynchronize(), "BFSKernel failed ",
			                __FILE__, __LINE__)) break;

		 	workset1.print();
		 	size = *workset1.sizep;
		 	printf("next size: %d\n", size);
		 }
		 ++curLevel;
	}
    
    

    levels.print();

    levels.del();
    visited.del();
    workset1.del();
	workset2.del();
	
}