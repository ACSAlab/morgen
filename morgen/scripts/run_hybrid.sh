#!/bin/bash

OUTDIR="run_hybrid"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2 kron
do
	echo ./bin/test $graph hybrid --block_size=256 --theta=2048 
	./bin/test $graph hybrid --block_size=256 --theta=2048 | grep 'Time' >> ./$OUTDIR.out
done