#!/bin/bash

OUTDIR="static_mapping_of_top_on_"




for i in 32 16 8 4 2 1
do
	echo $i >> ./$OUTDIR$1.out

	echo ./bin/test $1 topo --group_size=$i --instrument
	./bin/test $1 topo --group_size=$i --instrument  >> ./$OUTDIR$1.out

	echo -e '\n' >> ./$OUTDIR$1.out
done





