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
struct Metrics {

    int dedicatedLanes;
    int intraWastedLanes;
    int interWastedLanes;

    Metrics() {
        dedicatedLanes = 0;
        interWastedLanes = 0;
        intraWastedLanes = 0;
    }

    void display() {
        printf("[metric] dedicated:%d\n", dedicatedLanes);
        printf("[metric] inter wasted:%d\n", interWastedLanes);
        printf("[metric] intra wasted:%d\n", intraWastedLanes);
        printf("[metric] URI: %.4f", (float) interWastedLanes / dedicatedLanes);
        printf("[metric] URA: %.4f", (float) intraWastedLanes / dedicatedLanes);

    }

    void count(const List<VertexId, SizeT, Value> &list,
               const graph::CsrGraph<VertexId, SizeT, Value> &g,
               int group_size) 
    {
        list.transfer_back();

        int enlisted_warps;
        if (*list.sizep % 32 == 0)
            enlisted_warps = *list.sizep / 32;
        else 
            enlisted_warps = *list.sizep / 32 + 1;

        // which means the last warp could have threads
        // exceeding the list.elems[]'s bound

        int group_per_warp = 32 / group_size;

        // go through the warps
        for (int i = 0; i < enlisted_warps; i++) {

            int work_amount[32];
            int max_work = 0;


            // go though the groups in each warp
            for (int j = 0; j < group_per_warp; j++) {
                int index = i * group_per_warp + j;

                // query of work amount of the node assign the group
                if (index < *list.sizep) {
                    VertexId node = list.elems[index];
                    SizeT start = g.row_offsets[node];
                    SizeT end = g.row_offsets[node+1];
                    SizeT edge_num = end - start;
                    work_amount[j] = edge_num;
                } else {  // out of bound
                    max_work[j] = 0;
                }
                if (edge_num > max_work) max_work = edge_num;
            }

            for (int j = 0; j < group_per_warp; j++) {
                interWastedLanes += ((max_work / group_size - work_amount[j] / group_size) * group_size);
                intraWastedLanes += (group_size - work_amount[j] % group_size);
            }

            dedicatedLanes += ((max_work / group_size) * group_size * group_per_warp );

        }
    }

};






} // Utils
} // Morgen