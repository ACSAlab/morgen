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


#include <morgen/utils/timing.cuh>
#include <morgen/utils/list.cuh>
#include <morgen/workset/queue.cuh>

#define INF -1



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
        levels[outNode] = curLevel;

        SizeT outEdgeFirst = row_offsets[outNode];
        SizeT outEdgeLast = row_offsets[outNode+1];


        for (SizeT edge = outEdgeFirst; edge < outEdgeLast; edge++) {

                VertexId inNode = column_indices[edge];

                // if not visited, vistit it & append to the workset
                if (visited[inNode] == 0) {
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
                     bool display_distribution,
                     bool display_workset)
{

    // To make better use of the workset, we create two.
    // Instead of creating a new one everytime in each BFS level,
    // we just expand vertices from one to another
    workset::Queue<VertexId, SizeT> workset1(g.n);
    workset::Queue<VertexId, SizeT> workset2(g.n);


    // Initalize auxiliary list
    util::List<Value, SizeT> levels(g.n);
    levels.all_to(INF);


    // visitation list: 0 for unvisited
    util::List<int, SizeT> visited(g.n);
    visited.all_to(0);


    // traverse from source node
    workset1.append(source);   
    visited.set(source, 1);
    SizeT worksetSize = 1;
    SizeT lastWorksetSize = 1;
    SizeT visitedNodes = 0;
    Value curLevel = 0;


    printf("serial bfs starts... \n");  
    if (!instrument)
        printf("level\tfrontier_size\ttime\n");

    float total_millis = 0.0;

    while (worksetSize > 0) {


        util::CpuTimer timer;
        timer.start();

        lastWorksetSize = worksetSize;

        if (curLevel % 2 == 0) {

            BFSCore(g.row_offsets,
                    g.column_indices,
                    workset1.elems,
                    workset1.sizep,
                    workset2.elems,
                    workset2.sizep,
                    levels.elems,
                    curLevel,     
                    visited.elems);

            worksetSize = workset2.size();

            // traverse workset set, and query the edge number, then print it
            if (display_distribution) {
                for (int i = 0; i < *workset2.sizep; i++) {
                    VertexId outNode = workset2.elems[i];
                    SizeT outEdgeFirst = g.row_offsets[outNode];
                    SizeT outEdgeLast = g.row_offsets[outNode+1];
                    SizeT edges = outEdgeLast - outEdgeFirst;
                    printf("%d\t", edges);
                }
                printf("\n");
            }

            if (display_workset) {
                workset2.print();
            }


        } else {

            BFSCore(g.row_offsets,
                    g.column_indices,
                    workset2.elems,
                    workset2.sizep,
                    workset1.elems,
                    workset1.sizep,
                    levels.elems,
                    curLevel,
                    visited.elems);

            worksetSize = workset1.size();

            // traverse workset set, and query the edge number, then print it
            if (display_distribution) {
                for (int i = 0; i < *workset1.sizep; i++) {
                    VertexId outNode = workset1.elems[i];
                    SizeT outEdgeFirst = g.row_offsets[outNode];
                    SizeT outEdgeLast = g.row_offsets[outNode+1];
                    SizeT edges = outEdgeLast - outEdgeFirst;
                    printf("%d\t", edges);
                }
                printf("\n");
            }

            if (display_workset) {
                workset1.print();
            }

        }

        timer.stop();

        // do not print the timing infos when printing distribution
        if (instrument) printf("%d\t%d\t%.12f\n", curLevel, lastWorksetSize, timer.elapsedMillis());
        total_millis += timer.elapsedMillis();
        curLevel += 1;
        visitedNodes += lastWorksetSize;
    }

    printf("serial bfs terminates\n");  
    float billion_edges_per_second = (float)g.m / total_millis / 1000000.0;
    printf("time(s): %f   speed(BE/s): %f\n", total_millis / 1000.0, billion_edges_per_second);

    printf("%d nodes has been visited\n", visitedNodes);

    levels.print_log();

    levels.del();
    visited.del();
    workset1.del();
    workset2.del();

}

} // BFS
} // Morgen