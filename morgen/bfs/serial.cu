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




namespace morgen {

namespace bfs {
/**  
 *  Serial BFS, written like GPU kernel
 */
template<typename VertexId, typename SizeT, typename Value>
void
BFSCore(SizeT     *row_offsets,
        VertexId  *column_indices,
        VertexId  *worksetFrom,
        SizeT     *sizeFrom,
        VertexId  *worksetTo,
        SizeT     *sizeTo,
        Value     *levels,
        Value     curLevel,
        int       *visited)
{
    *sizeTo = 0;

    for (SizeT i = 0; i < *sizeFrom; i++) {
        VertexId outNode = worksetFrom[i];
        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];


        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

            VertexId inNode = column_indices[edge];

                // if not visited, vistit it & append to the workset
            if (visited[inNode] == 0) {
                levels[inNode] = curLevel + 1;
                visited[inNode] = 1;
                worksetTo[*sizeTo] = inNode;
                *sizeTo += 1;
            }
        }
    }
}

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

    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Queue<VertexId, SizeT> workset[] = {
        workset::Queue<VertexId, SizeT>(g.n),
        workset::Queue<VertexId, SizeT>(g.n),
    };
 
    // use to select between two worksets
    // src:  workset[selector]
    // dest: workset[selector ^ 1]
    int selector = 0;

    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to((Value) MORGEN_INF);

    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);

    // traverse from source node
    workset[selector].init(source);   
    visited.set(source, 1);
    levels.set(source, 0);
    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 1;
    SizeT visitedNodes = 0;
    Value curLevel = 0;

    printf("Serial bfs starts... \n");  
    if (instrument) printf("level\tfrontier size\tnew frontier size\tout edges\ttime\n");

    float total_millis = 0.0;

    while (worksetSize > 0) {
        lastWorksetSize = worksetSize;

        // calculate the outgoing edges before start searching.
        SizeT outgoingEdges = 0;
        for (int i = 0; i < worksetSize; i++) {
            VertexId outNode = workset[selector].elems[i];
            SizeT outEdges = g.row_offsets[outNode+1] - g.row_offsets[outNode];
            outgoingEdges += outEdges;
        }


        util::CpuTimer timer;
        timer.start();

        BFSCore(
            g.row_offsets,
            g.column_indices,
            workset[selector].elems,
            workset[selector].sizep,
            workset[selector ^ 1].elems,
            workset[selector ^ 1].sizep,
            levels.elems,
            curLevel,     
            visited.elems);

        worksetSize = *(workset[selector ^ 1].sizep);

        timer.stop();

        // do not print the timing infos when printing distribution
        if (instrument) printf("%d\t%d\t%d\t%d\t%.12f\n", curLevel, lastWorksetSize, worksetSize, outgoingEdges, timer.elapsedMillis());
        total_millis += timer.elapsedMillis();
        curLevel += 1;
        visitedNodes += lastWorksetSize;

        // traverse workset set, and query the edge number, then print it
        if (display_distribution) {
            for (int i = 0; i < *workset[selector ^ 1].sizep; i++) {
                VertexId outNode = workset[selector ^ 1].elems[i];
                SizeT outDegree = g.row_offsets[outNode+1] - g.row_offsets[outNode];
                printf("%d\t", outDegree);
            }
            printf("\n");
        }


        selector = selector ^ 1;
    }

    printf("Serial bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("Time(s):\t%f\nSpeed(BE/s):\t%f\n", total_millis / 1000.0, billion_edges_per_second);
    printf("%d nodes has been visited\n", visitedNodes);


    levels.print_log(true);

    levels.del();
    visited.del();
    workset[0].del();
    workset[1].del();

}

} // BFS
} // Morgen