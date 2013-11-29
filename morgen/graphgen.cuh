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

#include "graph.cuh"

/**
 * Generate a tiny graph from a null graph
 */
template<typename VertexId, typename SizeT, typename Value>
void tinyGraphGen(graph<VertexId, SizeT, Value> &g)
{
    SizeT      n = 9;
    SizeT      m = 11;
    SizeT      r[] = {0, 2, 5, 5, 6, 8, 9, 9, 11, 11};
    VertexId   c[] = {1, 3, 0, 2, 4, 4, 5, 7, 8, 6, 8};
    Value      v[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    g.init(n, m);

    for (SizeT i = 0; i < n + 1; i++)
        g.row_offsets[i] = r[i];

    for (SizeT i = 0; i < m; i++) {
        g.column_indices[i] = c[i];
        g.costs[i] = v[i];
    }
}

/**
 * Generate a graph from a user-defined format file
 * it take a pointer to the opened file
 */
template<typename VertexId, typename SizeT, typename Value> 
void myGraphGen(FILE *fp, graph<VertexId, SizeT, Value> &g)
{
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

}

