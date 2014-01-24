#!/bin/bash

OUTDIR="topo_vs_queue"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazonrmat2 random2
do
	echo ./bin/test $graph queue
	./bin/test $graph queue | grep -E 'Graph|Time' >> ./$OUTDIR.out
	echo ./bin/test $graph topo 
	./bin/test $graph topo  | grep -E 'Graph|Time' >> ./$OUTDIR.out
done