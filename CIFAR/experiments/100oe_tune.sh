#!/bin/bash

source ~/new_begin.sh

python oe_tune.py cifar100 --model $1 #-c

