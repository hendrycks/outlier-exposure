#!/bin/bash

source ~/new_begin.sh

python test.py --method_name $1 --num_to_avg 10 > snapshots/$1_test.txt
# oe_scratch, baseline

