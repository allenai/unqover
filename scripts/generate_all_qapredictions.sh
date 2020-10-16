#!/bin/bash

set -e

MODEL=$1
MODEL_NAME=$2
for STYPE in gender ethnicity religion country
do
 ./scripts/generate_qapredictions.sh ${MODEL} ${MODEL_NAME} $STYPE
done



