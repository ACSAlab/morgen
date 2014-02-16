#!/bin/bash



OUTDIR="block_size_for_"


for j in 32 64 128 256 512 1024 
do
	for graph in $1
	do
		echo ./bin/test $graph queue --block_size=$j
		./bin/test $graph queue --block_size=$j | grep -E 'Time' >> ./$OUTDIR$1.out
	done
done