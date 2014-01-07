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

#pragma once

#include <time.h>


#include <morgen/graph/graph.cuh>

namespace morgen { 
namespace graph {
namespace gen {
/**
 * Generate a graph from a user-defined format file
 * it take a pointer to the opened file
 */
template<typename VertexId, typename SizeT, typename Value> 
int myGraphGen(
    FILE *fp,
    CsrGraph<VertexId, SizeT, Value> &g)
{
    time_t mark0 = time(NULL);
    printf("My graph gen...\n");

	// read the number of nodes
	SizeT  nodes; 
	fscanf(fp, "%u", &nodes);
	g.initRow(nodes);   

	SizeT  start, width;
	int i;	
    for (i=0; i<g.n; i++) {
    	fscanf(fp, "%u %u", &start, &width);
    	g.row_offsets[i] = start;
    }

    // util here, you get the complete CSR represenation!
    g.row_offsets[i] = start + width;

    // read the user-defined source node and skip it
    VertexId  source;
    fscanf(fp, "%u", &source);

    // read the number of edges
    SizeT  edges;
    fscanf(fp, "%u", &edges);
	g.initColumn(edges);

    VertexId  dest;
    Value     cost;
    for (int j=0; j<g.m; j++) {
    	fscanf(fp, "%u %u", &dest, &cost);
    	g.column_indices[j] = dest;
    	g.costs[j] = cost;
    }

    time_t mark1 = time(NULL);
    printf("Done parsing (%ds).\n", (int) (mark1 - mark0));

    return 0;
}


} // Gen
} // Graph
} // Morgen
