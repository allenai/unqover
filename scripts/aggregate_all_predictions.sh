#!/bin/bash

set -e

MODEL_NAME=$1
for STYPE in gender ethnicity religion country
do
 ./scripts/aggregate_predictions.sh ${MODEL_NAME} $STYPE
done
