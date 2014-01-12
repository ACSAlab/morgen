#!/bin/bash

OUTDIR="diff_group_size"

mkdir -p $OUTDIR

for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	for i in 32 16 8 4 2
	do
		echo ./bin/test $exe queue --warp_map --group_size=$i
		./bin/test $exe queue --warp_map --group_size=$i | grep Time > $OUTDIR/$exe.group_size.$i.out
	done
done