#!/bin/bash

source ~/new_begin.sh

python test.py --method_name cifar10_$1 --num_to_avg 10 > snapshots/cifar10_$1_test.txt
# allconv_oe_scratch, wrn_baseline, wrn_oe_tune

