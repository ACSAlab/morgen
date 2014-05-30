/*
 *   Graph Representation on GPU
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <vector>
#include <morgen/graph/coo_edge_tuple.cuh>
#include <morgen/utils/cuda_util.cuh>
#include <cuda_runtime_api.h>




namespace morgen {

namespace graph {



/**************************************************************************
 * The graph is represented as CSR format and resides in the pinned memory
 * all edges in the graph are directed
 * In this version(for bfs only), we do not consider the edge weight 
 **************************************************************************/
template<typename VertexId, typename SizeT, typename Value>
struct CooGraph {

    SizeT     n;
    SizeT     m;

    /* Host pointer */
    CooEdgeTuple<VertexId>  *elems;

    /* Device pointers */
    CooEdgeTuple<VertexId>  *d_elems;

    /**
     * Default constructor(do not allocate memory since the n/m is unknown)
     */
    CooGraph() : n(0), m(0), elems(NULL), d_elems(NULL) {}


    void init(SizeT nodes, SizeT edges) {
        n = nodes;
        m = edges;
        
        elems = (CooEdgeTuple<VertexId>*) malloc ( sizeof(CooEdgeTuple<VertexId>) * m );
        
        if (util::handleError(
            cudaMalloc(  (void **) &d_elems, sizeof(CooEdgeTuple<VertexId>) * m ), 
            "CooGraph: cudaMalloc(d_elems) failed", __FILE__, __LINE__))
            exit(1);

    }

   void fromCoo(
        CooEdgeTuple<VertexId> *coo, 
        SizeT coo_nodes, 
        SizeT coo_edges,
        bool ordered_rows = false)
    {
        printf("[g] Converting %d vertices, %d directed edges (%s tuples) to COO format... \n",
            coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
        
        time_t mark1 = time(NULL);
        init(coo_nodes, coo_edges);

        // iterate through the all the edges(m == coo_edges)
        for (SizeT edge = 0; edge < m; edge++) {
            elems[edge].col = coo[edge].col;
            elems[edge].row = coo[edge].row;
        }

        time_t mark2 = time(NULL);
        printf("[g] Done converting (%ds).\n", (int) (mark2 - mark1));
    }


    void transfer() {
        printf("[g] Transfering graph from host memory to device memory...\n");

        time_t mark1 = time(NULL);


        if (util::handleError(cudaMemcpy(d_elems, elems, sizeof(CooEdgeTuple<VertexId>) * m, cudaMemcpyHostToDevice), 
            "CooGraph: hostToDevice(elems) failed", __FILE__, __LINE__)) exit(1);


        time_t mark2 = time(NULL);

        printf("[g] Done transfering (%ds).\n", (int) (mark2 - mark1));


    }


    /**
     * Delete the graph 
     */
    void del() { 
        
        if (elems) {
            util::handleError(cudaFree(d_elems), "CooGraph: cudaFree(d_elems) failed", __FILE__, __LINE__);
            free(elems);

        }

        n = 0;
        m = 0;

      }


};


} // Graph
} // Morgen


