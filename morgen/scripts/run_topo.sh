#!/bin/bash

OUTDIR="run_topo"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $graph topo --block_size=256 --threshold=2048
	./bin/test $graph topo --block_size=256 --threshold=2048 | grep 'Time' >> ./$OUTDIR.out
done