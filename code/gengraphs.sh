#!/bin/bash

for k in {5,10,20,30,40,50,60,70,80,90}
do
    for i in {1..700}
    do
        j=$((300+$i))
        python sim.py $k > data/N100_BA/malicious_$k/output$j
    done
done
