/*
 *   COO Edge Turple
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


namespace morgen {
namespace graph {



/**
 * Uneighted Edge tuple like (node1, node2) 
 */
template<typename VertexId>
struct CooEdgeTuple
{
	VertexId  row;
 	VertexId  col;

 	CooEdgeTuple(VertexId row, VertexId col) : row(row), col(col) {}
};



/**
 * Comparator for sorting COO edge tuple in CSR representation 
 */
template<typename tupleT>
bool tupleCompare(tupleT elem1, tupleT elem2)
{
	if (elem1.row < elem2.row)
		return true;
	else
		return false;

}


 } // graph
 } // morgen