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
#include <morgen/utils/stats.cuh>

namespace morgen {

namespace workset {

/**************************************************************************
 * a topology-ware hash
 **************************************************************************/
template<typename VertexId, typename SizeT, typename Value>
struct TopoHash {

    SizeT   slot_num;
    SizeT   slot_size_max[32];
    SizeT   n;                // total size of the hash table


    Value   *elems;
    SizeT   *slot_sizes;      // the logical size will be changed on gpu 
    SizeT   *slot_offsets;    

    Value   *d_elems;
    SizeT   *d_slot_sizes;
    SizeT   *d_slot_offsets;

    TopoHash() : n(0), slot_num(0), elems(NULL), slot_sizes(NULL), slot_offsets(NULL), d_elems(NULL), d_slot_sizes(NULL), d_slot_offsets(NULL) {} 

    TopoHash(const util::Stats<VertexId, SizeT, Value> &stat) {  

        slot_num = stat.outDegreeMax;
        n = 0;

        for (int i = 0; i < slot_num; i++) {
            slot_size_max[i] = stat.outDegreeLog[i+1];  // note outDegreeLog begins with 2^-1
            n += stat.outDegreeLog[i+1];
        }

        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;
        if (util::handleError(cudaHostAlloc((void **)&slot_sizes, sizeof(SizeT) * slot_num, flags),
                            "TopoHash: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostAlloc((void **)&slot_offsets, sizeof(SizeT) * slot_num, flags),
                            "TopoHash: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostAlloc((void **)&elems, sizeof(SizeT) * n, flags),
                            "TopoHash: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
        

        // initalize
        SizeT cursor = 0;
        for (int i = 0; i < slot_num; i++) {
            slot_sizes[i] = 0;
            slot_offsets[i] = cursor;
            cursor += slot_size_max[i];
        }

        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                        "TopoHash: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_slot_sizes, (void *) slot_sizes, 0),
                        "TopoHash: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_slot_offsets, (void *) slot_offsets, 0),
                        "TopoHash: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
    }


    // add to slot 0 anyway
    int insert(Value key) {
        slot_sizes[0] += 1;
        SizeT pos = slot_offsets[0] + slot_sizes[0];
        elems[pos] = key;
        return 0;
    }

    int total_size() {
        SizeT size = 0;
        for (int i = 0; i < slot_num; i++) {
            size += slot_sizes[i];
        }
        return size;
    }

    void info() {
        printf("[thash] slot_num: %d\n", slot_num);
        printf("[thash] n: %d\n", n);
        for (int i = 0; i < slot_num; i++) {
            printf("[thash] slot_size_max[%d]: %d\n", i, slot_size_max[i]);
        }
    }

    void del() {
        if (elems) {
            util::handleError(cudaFreeHost(elems), "TopoHash: cudaFreeHost(elems) failed", __FILE__, __LINE__);
            elems = NULL;
        }

        if (slot_sizes) {
            util::handleError(cudaFreeHost(slot_sizes), "TopoHash: cudaFreeHost(slot_sizes) failed", __FILE__, __LINE__);
            slot_sizes = NULL;     
        }

        if (slot_offsets) {
            util::handleError(cudaFreeHost(slot_offsets), "TopoHash: cudaFreeHost(slot_offsets) failed", __FILE__, __LINE__);
            slot_offsets = NULL;
        }
        
        n = 0;
        slot_num = 0;
        for (int i=0; i < slot_num; i++) {
            slot_size_max[i] = 0;
        }
        
    }

    ~TopoHash() {
        del();
    }

 };

} // Workset
} // Morgen