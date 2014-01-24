/*
 *   Utils
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



namespace morgen {

namespace util {

/**************************************************************************
 * How effcient a mapping stratefy can utilize the execution units
 **************************************************************************/
template<typename VertexId, typename SizeT, typename Value>
float calculateUtilizingEfficiency(
    const graph::CsrGraph<VertexId, SizeT, Value> &g,
    int group_size) 
{

    int utilized_slots = 0;
    int dedicated_slots = 0;

    for (SizeT i = 0; i < g.n; i++) {
        SizeT outDegree = g.row_offsets[i+1] - g.row_offsets[i];
        // 32 grouped thread process a node of 76 neibors will
        // produce a utilized_slots of 76（equal to the work amount）
        // a dedicated_slots of 32 * 3 
        utilized_slots += outDegree;

        if (outDegree % group_size == 0)
            dedicated_slots += (outDegree / group_size * group_size );
        else
            dedicated_slots += ((outDegree / group_size + 1) * group_size );

    }

    float utilizing_efficency = (float) utilized_slots / dedicated_slots;
    return utilizing_efficency;
}



} // Utils
} // Morgen