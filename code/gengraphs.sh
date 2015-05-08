#!/bin/bash

for i in {1..2}
do
    j=$((100+$i))
    python sim.py 30 > data/N100_BA/malicious_30/output$j
done
