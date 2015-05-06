#!/bin/bash

for i in {1..100}
do
    j=$((0+$i))
    python sim.py > data/N100_BA/malicious_40/output$j
done
