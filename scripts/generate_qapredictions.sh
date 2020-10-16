#!/bin/bash

set -e

MODEL=$1
MODEL_NAME=$2
DATASET_TYPE=$3
GPU_ID=0


TYPE=slot_act_map
case "${DATASET_TYPE}" in
    "gender")
        SUBJ=mixed_gender
        SLOT=gender_noact
        ACT=occupation_rev1
       ;;
    "country" | "religion" | "ethnicity")
        SUBJ=${DATASET_TYPE}
        SLOT=${DATASET_TYPE}_noact
        ACT=biased_${DATASET_TYPE}
        ;;
    *)
        echo "Can not handle subject class: ${DATASET_TYPE}"
        exit 1
        ;;
esac

FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}

python3 -u -m qa_hf.predict --gpuid ${GPU_ID} \
    --hf_model ${MODEL} \
    --input ${FILE}.source.json --output ./data/${MODEL_NAME}_${DATASET_TYPE}.output.json

