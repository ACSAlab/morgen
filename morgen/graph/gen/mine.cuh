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


#include <morgen/graph/csr_graph.cuh>

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
    printf("[my] Generating graph...\n");

    // read the number of nodes
    long long ll_nodes; 
    fscanf(fp, "%lld", &ll_nodes);
    printf("[my] Vertices: %lld", ll_nodes);
    g.initRow(ll_nodes);   

    long long start, width;
    int i;
    for (i = 0; i < g.n; i++) {
        fscanf(fp, "%lld %lld", &start, &width);
        g.row_offsets[i] = start;
    }
    // i == g.n
    g.row_offsets[i] = start + width;

    // util here, you get the complete CSR represenation!

    // read the user-defined source node and skip it
    VertexId  source;
    fscanf(fp, "%u", &source);

    // read the number of edges
    long long  ll_edges;
    fscanf(fp, "%u", &ll_edges);
    printf("[my] Edges: %lld", ll_edges);
    g.initColumn(ll_edges);

    long long  dest, cost;
    for (int j = 0; j < g.m; j++) {
        fscanf(fp, "%lld %lld", &dest, &cost);
        g.column_indices[j] = dest;
        //g.costs[j] = cost;
    }

    time_t mark1 = time(NULL);
    printf("[my] Done parsing (%ds).\n", (int) (mark1 - mark0));

    return 0;
}


} // Gen
} // Graph
} // Morgen
