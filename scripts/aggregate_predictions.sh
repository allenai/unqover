#!/bin/bash

set -e

MODEL_NAME=$1
DATASET_TYPE=$2

case "$DATASET_TYPE" in
    gender)
        GROUP_BY="gender_act"
        ;;
    *)
        GROUP_BY="subj_act"
        ;;
esac

python3 analysis.py \
    --metrics subj_bias,pos_err,attr_err \
    --input ./data/${MODEL_NAME}_${DATASET_TYPE}.output.json \
    --group_by ${GROUP_BY} --verbose 1 | tee ./data/${MODEL_NAME}_${DATASET_TYPE}.log.txt

