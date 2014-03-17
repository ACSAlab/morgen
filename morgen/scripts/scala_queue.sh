#!/bin/bash

OUTDIR="run_scala_rmat_32_queue"


for i in 1 2 3 4 5 6 7 8 9 10
do
	echo ./bin/test path queue --warp_map --group_size=32 --path=/home/chengyichao/Snap-2.1/examples/graphgen/rmat.32.${i}Mv --instrument
	./bin/test path queue --warp_map --group_size=32 --path=/home/chengyichao/Snap-2.1/examples/graphgen/rmat.32.${i}Mv --instrument | grep 'Expand' >> ./$OUTDIR.out
done