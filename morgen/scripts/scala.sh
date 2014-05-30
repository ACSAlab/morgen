#!/bin/bash

OUTDIR="run_scala_random_32"


for i in 1 2 4 8 16 32
do
	echo ./bin/test path topo --path=/home/chengyichao/Snap-2.1/examples/graphgen/random.32.${i}M --threshold=1536 --iteration=10
	./bin/test path topo --path=/home/chengyichao/Snap-2.1/examples/graphgen/random.32.${i}M --threshold=1536 --iteration=10 | grep 'Time' >> ./$OUTDIR.out
done