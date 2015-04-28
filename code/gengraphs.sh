#!/bin/bash

for i in {1..20}
do
    python sim.py > data/N16_RT/malicious_60/output$i
done
