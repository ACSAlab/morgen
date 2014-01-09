/*
 *   Graph Generator
 *
 *   Copyright (C) 2013 by
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

/******************************************************************************
 * DIMACS Graph Construction Routines
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <time.h>


#include <morgen/graph/csr_graph.cuh>


namespace morgen {
namespace graph {
namespace gen {


/**
 * Loads a DIMACS-formatted CSR graph through the fp. 
 */
template<typename VertexId, typename Value, typename SizeT>
int cooGraphGen(
	FILE* fp,
	CsrGraph<VertexId, SizeT, Value> &csr_graph)
{ 
	typedef CooEdgeTuple<VertexId> EdgeTupleT;

	time_t mark0 = time(NULL);
	printf("Coo graph gen...\n");

	char        line[1024];          // remaind line buffer
	char        c;
	long long   ll_nodes, ll_edges;
	long long   ll_node1, ll_node2;
	SizeT       edges_read = -1;
	EdgeTupleT  *coo;


	ll_nodes = 0;
	ll_edges = 0;

	while ((c = fgetc(fp)) != EOF) {
		switch (c) {

		

		case ' ':
		case '\t':
			// white space
			break;


		default:
			// put the char back
			ungetc(c, fp);

			// read the first line(nodes/edges)
			if (edges_read == -1) {
				fscanf(fp, "%lld %lld", &ll_nodes, &ll_edges, line);
				printf("%d nodes, %d edges\n", ll_nodes, ll_edges);
				coo = (EdgeTupleT*) malloc(sizeof(EdgeTupleT) * ll_edges * 2);
				edges_read++;
			} else {
				// process next edge in edge list
				fscanf(fp, "%lld %lld", &ll_node1, &ll_node2, line);
				coo[edges_read] = EdgeTupleT(ll_node1, ll_node2);
				edges_read++;
			}
		} // end of switch
	}

	if (ll_nodes == 0 || ll_edges == 0) {
		fprintf(stderr, "Error: nodes=%d edges=%d\n", ll_nodes, ll_edges);
		return 1;
	}

	edges_read--;

	// whether read_edges == the claimed edge numbers
	if (edges_read != ll_edges) {
		fprintf(stderr, "Error: only %d edges read(should be %d)\n", edges_read, ll_edges);
		return 1;
	}

	csr_graph.initFromCoo(coo, ll_nodes, ll_edges);

	time_t mark1 = time(NULL);
    printf("Done parsing (%ds).\n", (int) (mark1 - mark0));

    if (coo) free(coo);
	return 0;
}

} // namespace gem
} // namespace graph
} // namespace b40c
