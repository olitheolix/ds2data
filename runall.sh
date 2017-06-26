#!/bin/bash

# Run all the simulations for the Simple model and produce the figures in the blog.

set -e

# Colour
time python train.py --width=32 --height=32 --colour=rgb  --keep-model=0.9 --keep-spt=0.9 --num-dense=128 --num-sptr=0  --num-epochs=1000 --seed=0
python validate.py
echo '------------------------------------'

# Gray, large dense, with spatial transformer.
time python train.py --width=32 --height=32 --colour=gray --keep-model=0.9 --keep-spt=0.9 --num-dense=128 --num-sptr=20 --num-epochs=5000 --seed=0
python validate.py
echo '------------------------------------'

# Gray, small dense, without transformer.
time python train.py --width=32 --height=32 --colour=gray --keep-model=0.9 --keep-spt=0.9 --num-dense=32  --num-sptr=0  --num-epochs=3000 --seed=0
python validate.py
echo '------------------------------------'

# Gray, large dense, without transformer.
time python train.py --width=32 --height=32 --colour=gray --keep-model=0.9 --keep-spt=0.9 --num-dense=128 --num-sptr=0  --num-epochs=1500 --seed=0
python validate.py
echo '------------------------------------'

# Gray, small dense, with spatial transformer.
time python train.py --width=32 --height=32 --colour=gray --keep-model=0.9 --keep-spt=0.9 --num-dense=32  --num-sptr=20 --num-epochs=15000 --seed=0
python validate.py
echo '------------------------------------'
