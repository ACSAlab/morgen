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
struct Hash {

    SizeT   slot_num;
    SizeT   slot_size_max[32];
    SizeT   n;                // total size of the hash table
    bool    topo_hashed;   

    Value   *elems;
    SizeT   *slot_sizes;      // the logical size will be changed on gpu 
    SizeT   *slot_offsets;    

    Value   *d_elems;
    SizeT   *d_slot_sizes;
    SizeT   *d_slot_offsets;

    Hash() : n(0), slot_num(0), elems(NULL), slot_sizes(NULL), slot_offsets(NULL), d_elems(NULL), d_slot_sizes(NULL), d_slot_offsets(NULL) {} 


    Hash(const util::Stats<VertexId, SizeT, Value> &stat) {
        
        // calculate n and slot_num
        slot_num = stat.outDegreeMax;
        n = 0;
        for (int i = 0; i < slot_num; i++) {
            slot_size_max[i] = stat.outDegreeLog[i+1];  // note outDegreeLog begins with 2^-1
            n += stat.outDegreeLog[i+1];
        }
        printf("[hash] slot_num: %d\n", slot_num);
        printf("[hash] n: %d\n", n);
        for (int i = 0; i < slot_num; i++) {
            printf("[hash] slot_size_max[%d]: %d\n", i, slot_size_max[i]);
        }

        // use n and slot_num to allocate slot_sizes[] and slot_offsets
        slot_sizes = (SizeT*) malloc( sizeof(SizeT) * slot_num );
        slot_offsets = (SizeT*) malloc( sizeof(SizeT) * slot_num );
        elems = (Value*) malloc( sizeof(Value) * n );

        // set up slot_sizes and slot_offsets
        SizeT cursor = 0;
        for (int i = 0; i < slot_num; i++) {
            slot_sizes[i] = 0;
            slot_offsets[i] = cursor;
            cursor += slot_size_max[i];
        }

        topo_hashed = true;

        gpu_stuff();

    }

    // an ordonary hash
    Hash(SizeT _n, SizeT _s) {

        // calculate n and slot_num
        slot_num = _s;
        int each_slot_size = _n / slot_num + 1;
        n = each_slot_size * slot_num;

        // use n and slot_num to allocate slot_sizes[] and slot_offsets
        slot_sizes = (SizeT*) malloc( sizeof(SizeT) * slot_num );
        slot_offsets = (SizeT*) malloc( sizeof(SizeT) * slot_num );
        elems = (Value*) malloc( sizeof(Value) * n );


        // set up slot_sizes and slot_offsets
        for (int i = 0; i < slot_num; i++) {
            slot_sizes[i] = 0;
            slot_offsets[i] = i * each_slot_size;
        }

        topo_hashed = false;
        gpu_stuff();

    }

    void gpu_stuff() {
         if (util::handleError(cudaMalloc((void **) &d_slot_sizes, sizeof(SizeT) * slot_num),
            "Hash: cudaMalloc(d_slot_sizes) failed", __FILE__, __LINE__)) exit(1);

        if (util::handleError(cudaMalloc((void **) &d_slot_offsets, sizeof(SizeT) * slot_num),
            "Hash: cudaMalloc(d_slot_offsets) failed", __FILE__, __LINE__)) exit(1);

        if (util::handleError(cudaMalloc((void **) &d_elems, sizeof(Value) * n),
            "Hash: cudaMalloc(d_elems) failed", __FILE__, __LINE__)) exit(1);

        // transfer
        if (util::handleError(cudaMemcpy(d_slot_sizes, slot_sizes, sizeof(SizeT) * slot_num, cudaMemcpyHostToDevice), 
            "Hash: hostToDevice(slot_sizes) failed", __FILE__, __LINE__)) exit(1);

        if (util::handleError(cudaMemcpy(d_slot_offsets, slot_offsets, sizeof(SizeT) * slot_num, cudaMemcpyHostToDevice), 
            "Hash: hostToDevice(slot_offsets) failed", __FILE__, __LINE__)) exit(1);
    }

    void insert(Value key) {

        SizeT hash = topo_hashed ? 0 : (key % slot_num);
  
        // get slot_size[hash]
        if (util::handleError(cudaMemcpy(slot_sizes + hash, d_slot_sizes + hash, sizeof(SizeT) * 1, cudaMemcpyDeviceToHost), 
            "List: DeviceToHost(slot_sizes) failed", __FILE__, __LINE__)) exit(1);

        // get old value then increase
        SizeT old_size = slot_sizes[hash];
        slot_sizes[hash] += 1;  

        // update slot_size[hash]
        if (util::handleError(cudaMemcpy(d_slot_sizes + hash, slot_sizes + hash, sizeof(SizeT) * 1, cudaMemcpyHostToDevice), 
            "NaiveHash: hostToDevice(slot_offsets) failed", __FILE__, __LINE__)) exit(1);

        // slot_offsets won't change
        SizeT pos = slot_offsets[hash] + old_size;
        elems[pos] = key;

        // write element
        if (util::handleError(cudaMemcpy(d_elems + pos, elems + pos, sizeof(Value) * 1, cudaMemcpyHostToDevice), 
            "NaiveHash: hostToDevice(elems) failed", __FILE__, __LINE__)) exit(1);

    }


    // get the largest slot in the hash table
    int max_slot_size() {

        if (util::handleError(cudaMemcpy(slot_sizes, d_slot_sizes , sizeof(SizeT) * slot_num, cudaMemcpyDeviceToHost), 
            "Hash: DeviceToHost(slot_sizes) failed", __FILE__, __LINE__)) exit(1);

        SizeT  size = 0;
        for (int i = 0 ; i < slot_num; i++) {
            size = MORGEN_MAX(size, slot_sizes[i]);
        }
        return size;
    }

    // sum each slot size up
    int sum_slot_size() {

        if (util::handleError(cudaMemcpy(slot_sizes, d_slot_sizes , sizeof(SizeT) * slot_num, cudaMemcpyDeviceToHost), 
            "Hash: DeviceToHost(slot_sizes) failed", __FILE__, __LINE__)) exit(1);

        SizeT size = 0;
        for (int i = 0 ; i < slot_num; i++) {
            size += slot_sizes[i];
        }
        return size;
    }



    void del() {
        if (elems) {
            util::handleError(cudaFree(d_elems), "Hash: cudaFreeHost(d_elems) failed", __FILE__, __LINE__);
            elems = NULL;
        }

        if (slot_sizes) {
            util::handleError(cudaFree(d_slot_sizes), "Hash: cudaFreeHost(d_slot_sizes) failed", __FILE__, __LINE__);
            slot_sizes = NULL;     
        }

        if (slot_offsets) {
            util::handleError(cudaFree(d_slot_offsets), "Hash: cudaFreeHost(d_slot_offsets) failed", __FILE__, __LINE__);
            slot_offsets = NULL;
        }
        
        n = 0;
        slot_num = 0;
        for (int i=0; i < slot_num; i++) {
            slot_size_max[i] = 0;
        }
        
    }


 };

} // Workset
} // Morgen