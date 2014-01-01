
/*
 *   The breadth-first search algorithm(serial version)
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

#define INF -1


/**  
 *  Serial BFS 
 */
template<typename VertexId, typename SizeT, typename Value>
void
BFSCore(SizeT     *row_offsets,
 		VertexId  *column_indices,
 		VertexId  *worksetFrom,
 		SizeT     *sizeFrom,
 		VertexId  *worksetTo,
 		SizeT     *sizeTo,
 		Value     *levels,
 		Value     curLevel,
 		int       *visited)
{
	*sizeTo = 0;

	for (SizeT i = 0; i < *sizeFrom; i++) {
		VertexId outNode = worksetFrom[i];
		levels[outNode] = curLevel;

		SizeT outEdgeFirst = row_offsets[outNode];
		SizeT outEdgeLast = row_offsets[outNode+1];


		for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

				VertexId inNode = column_indices[edge];

				// if not visited, vistit it & append to the workset
				if (visited[inNode] == 0) {
					visited[inNode] = 1;
					worksetTo[*sizeTo] = inNode;
					*sizeTo += 1;
				}
		}
	}
}



template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_serial(graph<VertexId, SizeT, Value> &g, VertexId source)
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
    visited.set(source, 1);
    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 1;
	Value curLevel = 0;


    printf("serial bfs starts\n");	

    while (worksetSize > 0) {

		lastWorksetSize = worksetSize;

		if (curLevel % 2 == 0) {

			BFSCore(g.row_offsets,
				    g.column_indices,
				    workset1.elems,
				    workset1.sizep,
				    workset2.elems,
				    workset2.sizep,
				    levels.elems,
				    curLevel,     
				    visited.elems);

			worksetSize = workset2.size();

		} else {

			BFSCore(g.row_offsets,
		 		    g.column_indices,
		 		    workset2.elems,
		 		    workset2.sizep,
		 		    workset1.elems,
		 		    workset1.sizep,
		 		    levels.elems,
		 		    curLevel,
		 		    visited.elems);

			worksetSize = workset1.size();

		}
		printf("%d\t%d\n", curLevel, lastWorksetSize);
		curLevel += 1;
	}

	printf("serial bfs terminates\n");	
    levels.del();
    visited.del();
    workset1.del();
	workset2.del();

}