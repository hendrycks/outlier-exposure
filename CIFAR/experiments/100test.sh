#!/bin/bash

source ~/new_begin.sh

python test.py --method_name cifar100_$1 --num_to_avg 10 > snapshots/cifar100_$1_test.txt
# allconv_oe_scratch, wrn_baseline, wrn_oe_tune

