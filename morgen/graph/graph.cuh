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

#include <morgen/utils/cuda_util.cuh>
#include <cuda_runtime_api.h>




namespace morgen {

namespace graph {



/**************************************************************************
 * The graph is represented as CSR format and resides in the pinned memory
 * all edges in the graph are directed
 **************************************************************************/
template<typename VertexId, typename SizeT, typename Value>
struct CsrGraph {

    SizeT     n;
    SizeT     m;

    /**
     * Host pointers
     */
    SizeT     *row_offsets;
    VertexId  *column_indices;   // SOA instead of AOS 
    Value     *costs;

    /**
     * Device pointers
     */
    SizeT     *d_row_offsets;
    VertexId  *d_column_indices;
    Value     *d_costs;


    /**
     * Default constructor(do not allocate memory since the n/m is unknown)
     */
    CsrGraph() : n(0), m(0), row_offsets(NULL), column_indices(NULL), costs(NULL),
              d_row_offsets(NULL), d_column_indices(NULL), d_costs(NULL) {}



    void init(SizeT nodes, SizeT edges) {
        initRow(nodes);
        initColumn(edges); 
    }

    /**
     * Outgoing edge information is stored in row_offsets
     */
    void initRow(SizeT nodes) {
        n = nodes;

        // Allocated in the pinned memory
        // NOTE: the graph is mapped to device memory as well, the pointer
        // of which can be obtained by calling cudaHostGetDevicePointer()

        int flags = cudaHostAllocMapped;
        if (util::handleError(cudaHostAlloc((void **) &row_offsets, sizeof(SizeT) * (n + 1), flags),
                               "CsrGraph: cudaHostAlloc(row_offsets) failed", __FILE__, __LINE__)) 
            exit(1);

        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_row_offsets, (void *) row_offsets, 0),
                               "CsrGraph: cudaHostGetDevicePointer(d_row_offsets) failed", __FILE__, __LINE__)) 
            exit(1);
    }

    /**
     * In CSR format, column_indices is relative to edge size
     */
    void initColumn(SizeT edges) {
        m = edges;
        
        // Allocated in the pinned memory
        int flags = cudaHostAllocMapped;
        if (util::handleError(cudaHostAlloc((void **) &column_indices, sizeof(VertexId) * m, flags),
                               "CsrGraph: cudaHostAlloc(column_indices) failed", __FILE__, __LINE__)) 
            exit(1);
        if (util::handleError(cudaHostAlloc((void **) &costs, sizeof(Value) * m, flags),
                               "CsrGraph: cudaHostAlloc(costs) failed", __FILE__, __LINE__)) 
            exit(1);
        
        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_column_indices, (void *) column_indices, 0),
                               "CsrGraph: cudaHostGetDevicePointer(d_column_indices) failed", __FILE__, __LINE__)) 
            exit(1);
                   
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_costs, (void *) costs, 0),
                               "CsrGraph: cudaHostGetDevicePointer(d_costs) failed", __FILE__, __LINE__)) 
            exit(1);
    }


    /**
     * Delete the graph 
     */
    void del() { 
        
        util::handleError(cudaFreeHost(row_offsets), "CsrGraph: cudaFreeHost(row_offsets) failed",
                           __FILE__, __LINE__);
        util::handleError(cudaFreeHost(column_indices), "CsrGraph: cudaFreeHost(column_indices) failed",
                           __FILE__, __LINE__);
        util::handleError(cudaFreeHost(costs), "CsrGraph: cudaFreeHost(costs) failed",
                           __FILE__, __LINE__);
        
        n = 0;
        m = 0;
        row_offsets      = NULL;
        column_indices   = NULL;
        costs            = NULL;
        d_row_offsets    = NULL;
        d_column_indices = NULL;
        d_costs          = NULL;
      }


    ~CsrGraph() {
        del();
    }

    /**
    template <typename Tuple>
    void convertFromCoo(
        Tuple *coo, 
        SizeT coo_nodes, 
        SizeT coo_edges,
        bool ordered_rows = false)
    {
        printf("Converting %d vertices, %d directed edges (%s tuples) to CSR format... \n",
            coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");

        init(coo_nodes, coo_edges);

        // if unordered, sort it first
        if (!ordered_rows) {
            std::stable_sort(coo, coo + edges, tupleCompare<Tuple>);
        }

        VertexId prev_row = -1;

        // iterate through the all the edges
        for (SizeT edge = 0; edge < edges; edge++) {

            VertexId current_row = coo[edge].row;

            for (VertexId row = prev_row + 1; row <= current_row; row++) {
                row_offsets[row] = edge;
            }
            prev_row = current_row;

            column_indices[edge] = coo[edge].col;

            // weight of edge
            coo[edge].setVal(costs[edge]);
            
        }

        for (VertexId row = prev_row + 1; row < nodes; row++) {
            row_offsets[row] = edges;
        }
    }
    */



    
    /**
     * Display the infomation of the graph at the console
     */    
    void printInfo(bool verbose = false) {
        fprintf(stdout, "%lld vertices, %lld edges\n", (long long) n, (long long) m);

        if (!verbose) return;

        for (SizeT i = 0; i < n; i++) {
            printf("%lld", (long long)i);
            printf(" ->");
            for (SizeT j = row_offsets[i]; j < row_offsets[i+1]; j++) {
                printf(" ");
                printf("%lld", (long long)column_indices[j]);
                printf("(%lld)", (long long)costs[j]);                
            }
            printf("\n");
        }
    }


    /**
     * Count the outdegree of each node in log style
     * and display it in the console 
     */
    void printOutDegrees() {
        
        int log_counts[32];
        for (int i = 0; i < 32; i++) {
            log_counts[i] = 0;
        }
        
        int max_times = -1;

        for (SizeT i = 0; i < n; i++) {
            SizeT outDegree = row_offsets[i+1] - row_offsets[i];            
            int times = 0;
            while (outDegree > 0) {
                outDegree /= 2;  
                times++;                
            }
            if (times > max_times) max_times = times;
            log_counts[times]++;
        }
        
        for (int i = -1; i < max_times+1; i++) {
            int y = pow(2, i);
            printf("Degree %d:\t%d\t%.2f%%\n", y, log_counts[i+1],
                   (float) log_counts[i+1] * 100.0 / n);
        }

    }
};


} // Graph
} // Morgen


