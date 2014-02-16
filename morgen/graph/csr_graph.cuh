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
struct CsrGraph {

    SizeT     n;
    SizeT     m;

    /* Host pointer */
    SizeT     *row_offsets;
    VertexId  *column_indices;   // SOA instead of AOS 

    /* Device pointers */
    SizeT     *d_row_offsets;
    VertexId  *d_column_indices;



    /**
     * Default constructor(do not allocate memory since the n/m is unknown)
     */
    CsrGraph() : n(0), m(0), row_offsets(NULL), column_indices(NULL), d_row_offsets(NULL), d_column_indices(NULL) {}


    void init(SizeT nodes, SizeT edges) {
        initRow(nodes);
        initColumn(edges); 
    }

    /**
     * Outgoing edge information is stored in row_offsets
     */
    void initRow(SizeT nodes) {
        n = nodes;

        row_offsets = (SizeT*) malloc ( sizeof(SizeT) * (n + 1));

        if (util::handleError(cudaMalloc((void **) &d_row_offsets, sizeof(SizeT) * (n + 1)),
            "CsrGraph: cudaMalloc(d_row_offsets) failed", __FILE__, __LINE__)) exit(1);

    }


    /**
     * In CSR format, column_indices is relative to edge size
     */
    void initColumn(SizeT edges) {
        m = edges;
        
        column_indices = (VertexId*) malloc ( sizeof(VertexId) * m);

        if (util::handleError(cudaMalloc((void **) &d_column_indices, sizeof(VertexId) * m),
            "CsrGraph: cudaMalloc(d_column_indices) failed", __FILE__, __LINE__)) exit(1);

    }


    
    void initFromCoo(
        CooEdgeTuple<VertexId> *coo, 
        SizeT coo_nodes, 
        SizeT coo_edges,
        bool ordered_rows = false)
    {
        printf("[g] Converting %d vertices, %d directed edges (%s tuples) to CSR format... \n",
            coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
        
        time_t mark1 = time(NULL);

        init(coo_nodes, coo_edges);

        typedef CooEdgeTuple<VertexId> tupleT;
        // if unordered, sort it first
        if (!ordered_rows) {
            std::stable_sort(coo, coo + m, tupleCompare<tupleT>);
        }

        VertexId prev_row = -1;
        // iterate through the all the edges(m == coo_edges)
        for (SizeT edge = 0; edge < m; edge++) {

            VertexId current_row = coo[edge].row;

            // Fill up the missing rows
            for (VertexId row = prev_row + 1; row <= current_row; row++) {
                row_offsets[row] = edge;
            }

            prev_row = current_row;

            // It is a one-to-to map between ordered COO and CSR
            column_indices[edge] = coo[edge].col;

        }

        // Fill out any trailing rows that didn't have explicit lines in the file
        for (VertexId row = prev_row + 1; row <= n; row++) {
            row_offsets[row] = m;
        }


        time_t mark2 = time(NULL);

        printf("[g] Done converting (%ds).\n", (int) (mark2 - mark1));
    }
    


    void transfer() {
        printf("[g] Transfering graph from host memory to device memory...\n");

        time_t mark1 = time(NULL);


        if (util::handleError(cudaMemcpy(d_column_indices, column_indices, sizeof(VertexId) * m, cudaMemcpyHostToDevice), 
            "CsrGraph: hostToDevice(column_indices) failed", __FILE__, __LINE__)) exit(1);

        if (util::handleError(cudaMemcpy(d_row_offsets, row_offsets, sizeof(SizeT) * (n + 1), cudaMemcpyHostToDevice), 
            "CsrGraph: hostToDevice(row_offsets) failed", __FILE__, __LINE__)) exit(1);

        time_t mark2 = time(NULL);

        printf("[g] Done transfering (%ds).\n", (int) (mark2 - mark1));


    }


    /**
     * Delete the graph 
     */
    void del() { 
        
        if (row_offsets) {
            util::handleError(cudaFree(d_row_offsets), "CsrGraph: cudaFree(d_row_offsets) failed", __FILE__, __LINE__);
            free(row_offsets);

        }

        if (column_indices) {
            util::handleError(cudaFree(d_column_indices), "CsrGraph: cudaFree(d_column_indices) failed", __FILE__, __LINE__);
            free(column_indices);
        }
        
        n = 0;
        m = 0;

      }


};


} // Graph
} // Morgen


