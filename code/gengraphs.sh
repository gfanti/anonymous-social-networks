#!/bin/bash

for i in {1..100}
do
    python sim.py > data/N100_BA/malicious_60/output$i
done
