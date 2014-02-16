#!/bin/bash

OUTDIR="run_queue_warp_32"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat2 random2
do
	echo ./bin/test $graph queue --warp_map --group_size=32
	./bin/test $graph queue --warp_map --group_size=32 | grep -E 'Time' >> ./$OUTDIR.out
done