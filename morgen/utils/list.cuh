/*
 *   Data Structures Rrepresenting Workset
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


#include <morgen/utils/macros.cuh>


 namespace morgen {

 namespace util {

 
/******************************************************************************
 * Auxiliary list
 ******************************************************************************/
template<typename Value, typename SizeT>
struct List
{
    SizeT   n;
    Value   *elems;
    Value   *d_elems;

    List() : n(0), elems(NULL), d_elems(NULL) {}

    List(SizeT _n) {

        n = _n;

        // mapped & pinned
        int flags = cudaHostAllocMapped;
        if (util::handleError(cudaHostAlloc((void **)&elems, sizeof(Value) * n, flags),
                               "List: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) 
            exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                                "List: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__))
            exit(1);
    }

    // setting to some value on CPU serially
    void all_to(Value x) {
        for (int i = 0; i < n; i++) {
            elems[i] = x;
        }
    }

    void print_log() {
        int inf_count = 0;

        FILE* log = fopen("log.txt", "w");

        for (int i = 0; i < n; i++) {
            if (elems[i] == MORGEN_INF)  inf_count++; 
            fprintf(log, "%lld\n", (long long)elems[i]);
        }
        fprintf(log, "\n");

        printf("%.2f%% is infinity\n", (float) inf_count / n * 100.0);

    }

    void set(SizeT i, Value x) { elems[i] = x; }


    void del() {
        if (elems) {
            util::handleError(cudaFreeHost(elems), "List: cudaFreeHost(elems) failed", __FILE__, __LINE__);
            elems = NULL;
        }
        n = 0;
    }

    ~List() {
        del();
    }
};


} // Workset

} // Morgen
