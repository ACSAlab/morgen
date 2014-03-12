#!/bin/bash

OUTDIR="run_scala_rmat_32"


for i in 1 2 3 4 5 6 7 8 9 10
do
	echo ./bin/test path hybrid --path=/home/chengyichao/Snap-2.1/examples/graphgen/rmat.32.${i}Mv --threshold=3548 --alpha=4 --theta=16384 --instrument
	./bin/test path hybrid --path=/home/chengyichao/Snap-2.1/examples/graphgen/rmat.32.${i}Mv --threshold=3548 --alpha=4 --theta=16384 --instrument | grep 'Expand' >> ./$OUTDIR.out
done