#!/bin/bash
s=0
for i in *.jpg
do
    echo $s
    echo $i
    mv $i $s.jpg
    s=$[s+1]
done
