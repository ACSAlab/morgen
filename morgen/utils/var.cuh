/*
 *   Utils
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

namespace util {

/**************************************************************************
 * Single variable on GPU
 **************************************************************************/
template<typename Value>
struct Var {

    Value  *elem;
    Value  *d_elem;      

    Var(Value v = 0) { 
        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;

        if (util::handleError(cudaHostAlloc((void **)&elem, sizeof(Value) * 1, flags),
                               "var: cudaHostAlloc(elem) failed", __FILE__, __LINE__)) 
            exit(1);

        *elem = v;

        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_elem, (void *) elem, 0),
                               "var: cudaHostGetDevicePointer(d_elem) failed", __FILE__, __LINE__))
            exit(1);
    }

    Value getVal() {
    	return *elem;
    }

    void set(Value v) {
    	*elem = v;
    }

    void del() {
        util::handleError(cudaFreeHost(elem), "var: cudaFreeHost(elem) failed",
                           __FILE__, __LINE__);
        elem = NULL;
        d_elem = NULL;
    }

    ~Var() {
        del();
    }
 };

} // Utils
} // Morgen