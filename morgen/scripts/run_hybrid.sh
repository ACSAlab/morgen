#!/bin/bash

OUTDIR="run_hybrid"

for graph in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $graph hybrid --block_size=256 --threshold=3548 --alpha=4 --theta=32768 --instrument
	./bin/test $graph hybrid --block_size=256 --threshold=3548 --alpha=4 --theta=32768 --instrument | grep 'Expand' >> ./$OUTDIR.out
done