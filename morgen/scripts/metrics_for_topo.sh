#!/bin/bash

OUTDIR="metrics_for_topo"


for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo $graph >> ./$OUTDIR.out

	echo ./bin/test $graph topo --metrics
	./bin/test $graph topo  --metrics | grep -E 'metric' >> ./$OUTDIR.out

	echo -e '\n' >> ./$OUTDIR.out
	
done

