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


#include <morgen/graph/graph.cuh>


namespace morgen {
namespace graph {
namespace gen {


/**
 * Loads a DIMACS-formatted CSR graph through the fp. 
 */
template<typename VertexId, typename Value, typename SizeT>
int dimacsGraphGen(FILE* fp, CsrGraph<VertexId, SizeT, Value> &csr_graph)
{ 

	time_t mark0 = time(NULL);
	printf("Dimacs graph gen...\n");

	char        line[1024];          // remaind line buffer
	char	    c;                   // read in char
	SizeT	    edges_read = 0;      // how many edges in a row
	VertexId    current_node = -1;   // current out node we are checking
	long long   ll_edge;


	// parse the metis file
	while ((c = fgetc(fp)) != EOF) {
		switch (c) {

		case '%':
			// comment: skip any char encountered until see a '\n'
			while((c = fgetc(fp)) != EOF) {
				if (c == '\n') break; 
			}
			break;
		

		case ' ':
		case '\t':
			// white space
			break;

		case '\n':
			// end of line: begin to process the next node
			current_node++;
			csr_graph.row_offsets[current_node] = edges_read;
			break;

		default:
			// put the char back
			ungetc(c, fp);

			// read the first line(nodes/edges)
			if (current_node == -1) {
				long long ll_nodes, ll_edges;
				fscanf(fp, "%lld %lld[^\n]", &ll_nodes, &ll_edges, line);
				csr_graph.init(ll_nodes, ll_edges * 2);
				printf("%d nodes, %d directed edges\n", csr_graph.n, csr_graph.m);
			} else {
				// process next edge in edge list
				fscanf(fp, "%lld", &ll_edge);
				csr_graph.column_indices[edges_read] = ll_edge - 1;
				edges_read++;
			}
		} // end of switch
	}

	// Fill out any trailing rows that didn't have explicit lines in the file
	while (current_node < csr_graph.n) {
		current_node++;
		csr_graph.row_offsets[current_node] = edges_read;
	}

	// whether read_edges == the claimed edge numbers
	if (edges_read != csr_graph.m) {
		fprintf(stderr, "Error: only %d edges read\n", edges_read);
		csr_graph.del();
		return 1;
	}

	time_t mark1 = time(NULL);
    printf("Done parsing (%ds).\n", (int) (mark1 - mark0));

	return 0;
}

} // namespace gem
} // namespace graph
} // namespace b40c
