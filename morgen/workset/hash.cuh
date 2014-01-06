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

namespace workset {

/**************************************************************************
 * naive hash
 **************************************************************************/
template<typename Value, typename SizeT>
struct NaiveHash {

    SizeT   slot_num;
    SizeT   n;                // size in all
    SizeT   each_slot_size;

    Value  	*elems;
    SizeT   *slot_sizes;      //the logical size will be changed on gpu 
    SizeT   *slot_offsets;    

    Value   *d_elems;
    SizeT   *d_slot_sizes;
    SizeT   *d_slot_offsets;

    NaiveHash() : n(0), slot_num(0), elems(NULL), slot_sizes(NULL), slot_offsets(NULL), d_elems(NULL), d_slot_sizes(NULL), d_slot_offsets(NULL) {} 

    NaiveHash(SizeT _n, SizeT s_num) {  

        slot_num = s_num;

        // e.g. 7 elements will fit into 4 slots
        // each slots has 2 elements
    	each_slot_size = _n / slot_num + 1;
    	n = each_slot_size * slot_num;

        // Pinned and mapped in memory
        int flags = cudaHostAllocMapped;

        for (int i = 0; i < slot_num; i++) {
        	if (util::handleError(cudaHostAlloc((void **)&slot_sizes, sizeof(SizeT) * slot_num, flags),
          	                "NaiveHash: cudaHostAlloc(elems) failed", __FILE__, __LINE__)) exit(1);
        	if (util::handleError(cudaHostAlloc((void **)&slot_offsets, sizeof(SizeT) * slot_num, flags),
                            "NaiveHash: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
        	if (util::handleError(cudaHostAlloc((void **)&elems, sizeof(SizeT) * n, flags),
                            "NaiveHash: cudaHostAlloc(sizep) failed", __FILE__, __LINE__)) exit(1);
    	}

    	// initalize
    	for (int i = 0; i < slot_num; i++) {
    		slot_sizes[i] = 0;
    		slot_offsets[i] = i * each_slot_size;
    	}

        // Get the device pointer
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_elems, (void *) elems, 0),
                        "NaiveHash: cudaHostGetDevicePointer(d_elems) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_slot_sizes, (void *) slot_sizes, 0),
                        "NaiveHash: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
        if (util::handleError(cudaHostGetDevicePointer((void **) &d_slot_offsets, (void *) slot_offsets, 0),
                        "NaiveHash: cudaHostGetDevicePointer(d_sizep) failed", __FILE__, __LINE__)) exit(1);
    }


	// insert on cpu end
    int insert(Value key) {
    	SizeT hash = key % slot_num;
    	slot_sizes[hash] += 1;  // increase before writing
    	SizeT pos = slot_offsets[hash] + slot_sizes[hash];
    	elems[pos] = key;
    	return 0;  // succeed
    }

    // get the largest slot in the hash table
    int max_slot_size() {
    	SizeT  logical_size = 0;
    	for (int i = 0 ; i < slot_num; i++) {
    		logical_size = MORGEN_MAX(logical_size, slot_sizes[i]);
    	}
    	return logical_size;
    }

    // sum each slot size up
    int sum_slot_size() {
    	SizeT  logical_size = 0;
    	for (int i = 0 ; i < slot_num; i++) {
    		logical_size += slot_sizes[i];
    	}
    	return logical_size;
    }

    void del() {
        util::handleError(cudaFreeHost(elems), "NaiveHash: cudaFreeHost(elems) failed", __FILE__, __LINE__);
        util::handleError(cudaFreeHost(slot_sizes), "NaiveHash: cudaFreeHost(slot_sizes) failed", __FILE__, __LINE__);
        util::handleError(cudaFreeHost(slot_offsets), "NaiveHash: cudaFreeHost(slot_offsets) failed", __FILE__, __LINE__);
        
        n = 0;
        slot_num = 0;
        each_slot_size = 0;
        
        elems = NULL;
        slot_sizes = NULL;     
        slot_offsets = NULL;  
        d_elems = NULL;
        d_slot_sizes = NULL;     
        d_slot_offsets = NULL; 
    }

    ~NaiveHash() {
        del();
    }

 };

} // Workset
} // Morgen