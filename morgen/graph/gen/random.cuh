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
 * Builds a random CSR graph by adding edges edges to nodes nodes by randomly choosing
 * a pair of nodes for each edge.  There are possibilities of loops and multiple 
 * edges between pairs of nodes.    
 * 
 * Returns 0 on success, 1 on failure.
 */
template<typename VertexId, typename SizeT, typename Value> 
int randomGraphGen(
    SizeT nodes,
    SizeT edges,
    CsrGraph<VertexId, SizeT, Value> &csr_garph,
    bool undirected)
{

    if ((nodes < 0) || (edges < 0)) {
        fprintf(stderr, "Invalid graph size: nodes=%d, edges=%d", nodes, edges);
        return 1;
    }

    printf("  Selecting %llu %s random edges in COO format...\n", 
        (unsigned long long) edges, (undirected) ? "undirected" : "directed");


    typedef CooEdgeTuple<VertexId> EdgeTupleT;
    // a->b  b->a : 2 directed edges
    SizeT directed_edges = (undirected) ? edges * 2 : edges;
    EdgeTupleT *coo = (EdgeTupleT*) malloc(sizeof(EdgeTupleT) * directed_edges);

    for (SizeT i=0; i < edges; i++) {
        coo[i].row = randomNode(nodes);
        coo[i].col = randomNode(nodes);
        coo[i].val = 1;
        if (undirected) {
            // reverse edge
            coo[edges + i].row = coo[i].col;
            coo[edges + i].col = coo[i].row;
        }
    }

    // convert from COO to CSR
    g.convertFromCoo(coo, nodes, directed_edges);

    free(coo);
    return 0;
}


} // Gen
} // Graph
} // Morgen
