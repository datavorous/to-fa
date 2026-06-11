#!/bin/bash
# Usage: ./sync.sh <user>@turing.iiit.ac.in
# Example: ./sync.sh sagnik.bhattacharjee@turing.iiit.ac.in
REMOTE=${1:?"usage: ./sync.sh <user@host>"}
rsync -av --mkpath \
  --exclude='.venv' --exclude='results' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='*.egg-info' --exclude='sync.sh' \
  /home/datavorous/research/gemma-finetune/inf-serving/ \
  "$REMOTE:~/to-fa/"
