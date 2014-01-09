/*
 *   Graph Gen util
 *
 *   Copyright (C) 2013 by
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

/**
 * This piece of code is copyed from back40computing
 */
 
template <typename K>
void randomBits(K &key, int entropy_reduction = 0, int lower_key_bits = sizeof(K) * 8)
{
    const unsigned int NUM_UCHARS = (sizeof(K) + sizeof(unsigned char) - 1) / sizeof(unsigned char);
    unsigned char key_bits[NUM_UCHARS];
    

    do {
    
        for (int j = 0; j < NUM_UCHARS; j++) {
            unsigned char quarterword = 0xff;
            for (int i = 0; i <= entropy_reduction; i++) {
                quarterword &= (rand() >> 7);
            }
            key_bits[j] = quarterword;
        }
        
        if (lower_key_bits < sizeof(K) * 8) {
            unsigned long long base = 0;
            memcpy(&base, key_bits, sizeof(K));
            base &= (1 << lower_key_bits) - 1;
            memcpy(key_bits, &base, sizeof(K));
        }
        
        memcpy(&key, key_bits, sizeof(K));
        
    } while (key != key);       // avoids NaNs when generating random floating point numbers 
}

template<typename SizeT>
 SizeT randomNode(SizeT num_nodes) {
    SizeT node_id;
    randomBits(node_id);
    if (node_id < 0) node_id *= -1;
    return node_id % num_nodes;
 }


} // Graph

} // Morgen