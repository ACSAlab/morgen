#!/bin/bash

OUTDIR="run_topo"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 random3 kron
do
	echo ./bin/test $graph topo --block_size=512  --iteration=10 --instrument
	./bin/test $graph topo --block_size=512 --iteration=10  --instrument | grep 'Expand' >> ./$OUTDIR.out
done