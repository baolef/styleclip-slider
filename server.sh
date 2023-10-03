#!/bin/bash

port=1225
device=0
config='configs/default.yaml'

while getopts p:d:c: flag; do
  case "${flag}" in
  p) port=${OPTARG} ;;
  d) device=${OPTARG} ;;
  c) config=${OPTARG} ;;
  esac
done

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate zoom
python demo.py --port $port --gpu $device --config $config
