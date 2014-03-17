/*
 *   Macros
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



#define MORGEN_MAX(a, b) ((a > b) ? a : b)

#define MORGEN_MIN(a, b) ((a < b) ? a : b)

#define MORGEN_BLOCK_NUM(a, b) ((a % b == 0 ? (a / b) : (a / b + 1)))

#define MORGEN_BLOCK_NUM_SAFE(a, b) MORGEN_MIN(MORGEN_BLOCK_NUM(a, b), 65536)

#define MORGEN_INF 2147483647






