/*
 *   The breadth-first search algorithm(serial version)
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
#include <morgen/utils/timing.cuh>
#include <morgen/utils/list.cuh>
#include <morgen/workset/queue.cuh>
#include <deque>




namespace morgen {

namespace bfs {


/**
 * Serial BFS is also used to inspect the workload distribution
 * within the froniter.
 */
template<typename VertexId, typename SizeT, typename Value>
void BFSGraph_serial(const graph::CsrGraph<VertexId, SizeT, Value> &g,
                     VertexId source,
                     bool instrument,
                     bool display_distribution)
{


    std::deque<VertexId> current;
    //std::deque<VertexId> next;
 


    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);



    current.push_back(source);
    visited.set(source, 1);
    levels.set(source, 0);

    printf("Serial bfs starts... \n");  

    float total_millis = 0.0;

    util::CpuTimer timer;
    timer.start();


    while(!current.empty()) {


        VertexId outNode = current.front();
        current.pop_front();


        SizeT outEdgeFirst = g.row_offsets[outNode];
        SizeT outEdgeLast = g.row_offsets[outNode+1];

        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge ++) {

            VertexId inNode = g.column_indices[edge];
            if (visited.elems[inNode] == 0) {
                levels.elems[inNode] = levels.elems[outNode] + 1; 
                current.push_back(inNode);
                visited.elems[inNode] = 1;
            }
        }


        visited.elems[outNode] = 2;

    }



    timer.stop();
    total_millis = timer.elapsedMillis();

    printf("Serial bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);
    //printf("%d nodes has been visited\n", visitedNodes);


    levels.print_log(true);

    levels.del();
    visited.del();


}

} // BFS
} // Morgen