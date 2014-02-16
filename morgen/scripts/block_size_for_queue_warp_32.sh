#!/bin/bash

OUTDIR="block_size_for_queue_warp_32"



for j in 32 64 128 256 512 1024
do
	for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
	do
		echo ./bin/test $graph queue --warp_map --group_size=32 --block_size=$j
		./bin/test $graph queue --warp_map --group_size=32 --block_size=$j | grep -E 'Time' >> ./$OUTDIR.out
	done
done

