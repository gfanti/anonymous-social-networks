#!/bin/bash

for i in {1..100}
do
    j=$((100+$i))
    python sim.py > data/N100_BA/malicious_90/output$j
done
