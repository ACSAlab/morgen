#!/bin/bash

OUTDIR="unordered_ordered_work_amount"

for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $exe queue --warp_map --group_size=32
	./bin/test $exe queue --warp_map --group_size=32 | grep -E 'Graph|Unorderd|Time|Blocks' >> ./$OUTDIR.out
	echo ./bin/test $exe queue --unordered --warp_map --group_size=32
	./bin/test $exe queue --unordered --warp_map --group_size=32 | grep -E 'Graph|Unorderd|Time|Blocks' >> ./$OUTDIR.out
done