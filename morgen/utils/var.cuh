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

    Var() { 
        elem = (Value*) malloc( sizeof(Value) * 1);
        if (util::handleError(cudaMalloc((void **) &d_elem, sizeof(Value) * 1),
            "Var: cudaMalloc(d_elem) failed", __FILE__, __LINE__)) exit(1);

    }

    Value getVal() {
        if (util::handleError(cudaMemcpy(elem, d_elem, sizeof(Value) * 1, cudaMemcpyDeviceToHost), 
            "Var: DeviceToHost(elem) failed", __FILE__, __LINE__)) exit(1);
        return *elem;
    }

    void set(Value v) {
        *elem = v;
        if (util::handleError(cudaMemcpy(d_elem, elem, sizeof(Value) * 1, cudaMemcpyHostToDevice), 
            "Var: hostToDevice(elem) failed", __FILE__, __LINE__)) exit(1);
    }

    void del() {
        if (elem) {
            util::handleError(cudaFree(d_elem), "var: cudaFree(elem) failed", __FILE__, __LINE__);
            free(elem);
        }

    }


 };

} // Utils
} // Morgen