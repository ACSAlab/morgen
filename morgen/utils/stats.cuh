/*
 *   Stats
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


#include <morgen/utils/log.cuh>



namespace morgen {

namespace util {


template<typename VertexId, typename SizeT, typename Value>
struct Stats {

    SizeT vertices;
    SizeT edges;
    SizeT outDegreeLog[32];
    SizeT bucket[6];
    SizeT outDegreeMax;      
    SizeT totalDegree;
    Value min;
    Value quartile_first;
    Value median;
    Value quartile_second;
    Value max;

    void display() {
        printf("%d\n", outDegreeMax);
        printf("[stat] Vertices:\t%lld\n[stat] Edges:\t%lld\n", (long long) vertices, (long long) edges);
        printf("[stat] Avg. Outdegree:\t%.1f\n", (float) totalDegree / vertices);
        printf("[stat] Quartiles:\t%d\t%d\t%d\t%d\t%d\n", min, quartile_first, median, quartile_second, max); 

        // outDegreeLog[0]
        printf("[stat] outDegreeLog[0]: 0\t%d\t%.2f%%\n", outDegreeLog[0], (float) outDegreeLog[0] * 100.0 / vertices);

        // outDegreeLog[1] - outDegreeLog[max]
        for (int i = 1; i <= outDegreeMax; i++) {
            int high = pow(2, i-1);
            int low = pow(2, i-2);
            printf("[stat] outDegreeLog[%d]: (%d, %d]\t%d\t%.2f%%\n", i, low, high, outDegreeLog[i], (float) outDegreeLog[i] * 100.0 / vertices);
        }
    }

    void gen(const graph::CsrGraph<VertexId, SizeT, Value> &g) {
        vertices = g.n;
        edges = g.m;

        // count outdegree dristribution
        for (int i = 0; i < 32; i++) {
            outDegreeLog[i] = 0;
        }

        outDegreeMax = -1;

        for (SizeT i = 0; i < g.n; i++) {
            SizeT outDegree = g.row_offsets[i+1] - g.row_offsets[i];  

            int times = getLogOf(outDegree) + 1;  // getLogOf(outDegree) can be -1
            /*
            while (outDegree > 0) {
                outDegree /= 2;  
                times++;                
            }
            */

            if (times > outDegreeMax) 
                outDegreeMax = times;

            outDegreeLog[times]++;
        }

        // quatiles
        totalDegree = 0;
        std::vector<Value> outdegree_vec;

        for (int i = 0; i < g.n; i++) {
            SizeT outdegree = g.row_offsets[i+1] - g.row_offsets[i];  
            outdegree_vec.push_back(outdegree); 
            totalDegree += outdegree;          
        }

        // setting up bucket size according to the graph

        std::sort(outdegree_vec.begin(), outdegree_vec.end());
        min = outdegree_vec[0];
        quartile_first = outdegree_vec[vertices/4];
        median = outdegree_vec[vertices/2];
        quartile_second = outdegree_vec[vertices / 4 * 3];
        max = outdegree_vec[vertices-1];
    }

};






} // Utils
} // Morgen