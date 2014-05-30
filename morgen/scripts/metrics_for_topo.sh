#!/bin/bash

OUTDIR="metrics_for_topo"


for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 random3 kron
do
	echo $graph >> ./$OUTDIR.out

	echo ./bin/test $graph topo --instrument
	./bin/test $graph topo  --instrument | grep -E 'metric' >> ./$OUTDIR.out

	echo -e '\n' >> ./$OUTDIR.out
	
done

