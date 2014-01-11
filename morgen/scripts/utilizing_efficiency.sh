#!/bin/bash
for exe in mesh fla thermal eco audi copaper livejournal kkt amazon rmat1 rmat2 random1 random2
do
	echo ./bin/test $exe nil --metrics 
	./bin/test $exe nil --metrics 
done