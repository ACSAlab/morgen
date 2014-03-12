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
#include <math.h>


namespace morgen {

namespace util {


template<typename VertexId, typename SizeT, typename Value>
struct Metrics {

    int dedicatedLanes;
    int intraWastedLanes;
    int interWastedLanes;
    int utilizedLanes;

    Metrics() {
        dedicatedLanes = 0;
        interWastedLanes = 0;
        intraWastedLanes = 0;
        utilizedLanes = 0;
    }


    void clear() {
        dedicatedLanes = 0;
        interWastedLanes = 0;
        intraWastedLanes = 0;
        utilizedLanes = 0; 
    }

    void display() {
        /*
        printf("[metric] dedicated:\t%d\n", dedicatedLanes);
        printf("[metric] utilized:\t%d\n", utilizedLanes);
        printf("[metric] inter wasted:\t%d\n", interWastedLanes);
        printf("[metric] intra wasted:\t%d\n", intraWastedLanes);
        printf("[metric] URI:\t%.4f\n", (float) interWastedLanes / dedicatedLanes);
        printf("[metric] URA:\t%.4f\n", (float) intraWastedLanes / dedicatedLanes);
        printf("[metric] UR:\t%.4f\n", (float) utilizedLanes / dedicatedLanes);
        */
        printf("[metric] \t%.4f\t%.4f\t%.4f\n", (float) interWastedLanes / dedicatedLanes,
         (float) intraWastedLanes / dedicatedLanes,
          (float) utilizedLanes / dedicatedLanes);
    }

    void count(const VertexId* workset,
               SizeT workset_size,
               const graph::CsrGraph<VertexId, SizeT, Value> &g,
               int group_size) 
    {


        int group_per_warp = 32 / group_size;

        int enlisted_warps;
        if (workset_size % group_per_warp == 0)
            enlisted_warps = workset_size / group_per_warp;
        else 
            enlisted_warps = workset_size / group_per_warp + 1;

        // which means the last warp could have threads
        // exceeding the list.elems[]'s bound


        // go through the warps
        for (int i = 0; i < enlisted_warps; i++) {

            int work_amount[32] = {0};
            int max_work = 0;

            // go though the groups in each warp
            for (int j = 0; j < group_per_warp; j++) {
                int index = i * group_per_warp + j;

                // query of work amount of the node assign the group
                if (index < workset_size) {
                    VertexId node = workset[index];
                    SizeT start = g.row_offsets[node];
                    SizeT end = g.row_offsets[node+1];
                    SizeT edge_num = end - start;
                    work_amount[j] = edge_num;
                    if (edge_num > max_work) max_work = edge_num;
                    utilizedLanes += edge_num;
                } else {  // out of bound
                    work_amount[j] = 0;
                }
            }

            /*
            printf("warp: %d\n", i);
            for (int j = 0; j < group_per_warp; j++) {
                printf("%d  ", work_amount[j]);
            }
            printf("\n\n");
            */
            for (int j = 0; j < group_per_warp; j++) {

                int a = ((int) ceil( (float) max_work / group_size ) -
                         (int) ceil( (float) work_amount[j] / group_size )
                        ) * group_size;

                int b;

                if (work_amount[j] % group_size == 0) b = 0;
                else b = group_size - work_amount[j] % group_size;

                //printf("%d, %d\n", a, b);

                interWastedLanes += a;
                intraWastedLanes += b;

            }
            

            dedicatedLanes += ( (int) ceil( (float) max_work / group_size ) * group_size * group_per_warp );

        }
    }

};






} // Utils
} // Morgen