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
    
        elems = (Value*) malloc( sizeof(Value) * n);
        sizep = (SizeT*) malloc( sizeof(SizeT) * 1);

        *sizep = 0;

        if (util::handleError(cudaMalloc((void **) &d_elems, sizeof(Value) * n),
            "Queue: cudaMalloc(d_elems) failed", __FILE__, __LINE__)) exit(1);


        if (util::handleError(cudaMalloc((void **) &d_sizep, sizeof(SizeT) * 1),
            "Queue: cudaMalloc(d_sizep) failed", __FILE__, __LINE__)) exit(1);

    }


    /**
     * A.K.A. enqueue a value
     */
    void init(Value v) {

        *sizep = 1;
        elems[0] = v;

        if (util::handleError(cudaMemcpy(d_elems, elems, sizeof(Value) * 1, cudaMemcpyHostToDevice), 
            "List: hostToDevice(elems) failed", __FILE__, __LINE__)) exit(1);

        if (util::handleError(cudaMemcpy(d_sizep, sizep, sizeof(SizeT) * 1, cudaMemcpyHostToDevice), 
            "List: hostToDevice(sizep) failed", __FILE__, __LINE__)) exit(1);

    }


    int size() {

        if (util::handleError(cudaMemcpy(sizep, d_sizep, sizeof(SizeT) * 1, cudaMemcpyDeviceToHost), 
            "List: DeviceToHost(elems) failed", __FILE__, __LINE__)) exit(1);

        return *sizep;
    }

/*
    void print() {
        for (int i = 0; i < *sizep; i++) {
            printf("%lld\t", (long long)elems[i]);
        }
        printf("\n");
    }
*/


    void del() {
        
        if (elems) {
            util::handleError(cudaFree(elems), "Queue: cudaFree(elems) failed", __FILE__, __LINE__);
            free(elems);

        }

        if (sizep) {
            util::handleError(cudaFree(sizep), "Queue: cudaFree(sizep) failed", __FILE__, __LINE__);
            free(sizep);
        }

        n = 0;

    }


    ~Queue() {
        del();
    }
 };


} // Workset
} // Queue

