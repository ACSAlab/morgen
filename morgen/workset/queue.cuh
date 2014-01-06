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

namespace morgen {

namespace workset {

/**************************************************************************
 * Static queue
 **************************************************************************/
template<typename Value, typename SizeT>
struct Queue {

    SizeT     n;         //maximal allocated size
    Value     *elems;
    SizeT     *sizep;      //the logical size will be changed on gpu 

    Value     *d_elems;
    SizeT     *d_sizep;

    Queue() : n(0), sizep(NULL), elems(NULL), d_sizep(NULL), d_elems(NULL) {} 

    Queue(SizeT _n) { 

        n = _n;
    
        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;
        if (util::handleError(cudaHostAlloc((void **)&elems, sizeof(SizeT) * n, flags),
                               "Queue: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) 
            exit(1);
        if (util::handleError(cudaHostAlloc((void **)&sizep, sizeof(SizeT) * 1, flags),
                               "Queue: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) 
            exit(1);

        *sizep = 0;

        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                               "Queue: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__))
            exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_sizep, (void *) sizep, 0),
                               "Queue: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__))
            exit(1);

    }


    /**
     * A.K.A. enqueue
     */
    int append(Value v) {
        if (*sizep >= n)    // has been full
            return -1;
        else {
            elems[*sizep] = v; 
            *sizep += 1;
            return 0;
        }
    }


    int size() {
        if (sizep)  return *sizep;
        else        return -1;
    }

    void print() {
        for (int i = 0; i < *sizep; i++) {
            printf("%lld\n", (long long)elems[i]);
        }
        printf("\n");
    }

    void del() {
        util::handleError(cudaFreeHost(elems), "Queue: cudaFreeHost(elems) failed",
                           __FILE__, __LINE__);
        util::handleError(cudaFreeHost(sizep), "Queue: cudaFreeHost(sizep) failed",
                           __FILE__, __LINE__);
        n = 0;
        sizep = NULL;
        d_sizep = NULL;
        elems = NULL;
        d_elems = NULL;
    }


    ~Queue() {
        del();
    }
 };


} // Workset
} // Queue

