#!/bin/bash


GRAPH="/home/chengyichao/food/rmat.5Mv.60Me"

OPTIONS=""


echo ./bin/test $GRAPH mine serial 
     ./bin/test $GRAPH mine serial

echo ./bin/test $GRAPH mine bitmask 
     ./bin/test $GRAPH mine bitmask

echo ./bin/test $GRAPH mine queue 
     ./bin/test $GRAPH mine queue

echo ./bin/test $GRAPH mine hash --slots=4 
     ./bin/test $GRAPH mine hash --slots=4