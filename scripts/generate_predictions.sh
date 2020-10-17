#!/bin/bash

set -e

gpuid=${gpuid:-0}

if [ "$1" == "-h" ]; then
  echo "Generate predictions using all models and all datasets we have."
  echo "   --gpuid       The GPU device index to use, default to 0"
  echo "   -h           Print the help message and exit"
  exit 0
fi


while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

DATA=gender,country,ethnicity,religion
# mased LMs
./generate_lm_predictions.sh --m roberta-base --m_name robertabase --d $DATA --gpuid $gpuid
./generate_lm_predictions.sh --m roberta-large --m_name robertalarge --d $DATA --gpuid $gpuid
./generate_lm_predictions.sh --m distilbert-base-uncased --m_name distilbert --d $DATA --gpuid $gpuid
./generate_lm_predictions.sh --m bert-base-uncased --m_name bertbase --d $DATA --gpuid $gpuid
./generate_lm_predictions.sh --m bert-large-uncased-whole-word-masking --m_name bertlarge --d $DATA --gpuid $gpuid

# SQuAD models
#   for roberta family, we use the ones trained on our own
./generate_qa_predictions.sh --m ./models/squad_seqtok --m_name robertabase --t roberta-base --d $DATA --gpuid $gpuid
./generate_qa_predictions.sh --m ./models/squad_seqtok_robertalarge --m_name robertalarge --t roberta-large --d $DATA --gpuid $gpuid
#   for bert base and dilstilbert, use the ones trained on our own with HF's internfaces
./generate_qa_predictions_hf.sh --m ./models/bert-base-uncased-squad --m_name bertbase --d $DATA --gpuid $gpuid
./generate_qa_predictions_hf.sh --m ./models/distilbert-base-uncased-squad --m_name distilbert --d $DATA --gpuid $gpuid
#   for bertlarge, we use the officially released model from HF
./generate_qa_predictions_hf.sh --m bert-large-uncased-whole-word-masking-finetuned-squad --m_name bertlarge --d $DATA --gpuid $gpuid

# NewsQA models
#   here we use models trained on our own
./generate_qa_predictions.sh --m ./models/newsqa_seqtok --m_name newsqa_robertabase --t roberta-base --d  --gpuid $gpuid
./generate_qa_predictions.sh --m ./models/newsqa_seqtok_robertalarge --m_name newsqa_robertalarge --t roberta-large --d $DATA --gpuid $gpuid
./generate_qa_predictions.sh --m ./models/newsqa_seqtok_distilbert --m_name newsqa_distilbert --t distilbert-base-uncased --d $DATA --gpuid $gpuid
./generate_qa_predictions.sh --m ./models/newsqa_seqtok_bertbase --m_name newsqa_bertbase --t bert-base-uncased --d $DATA --gpuid $gpuid
./generate_qa_predictions.sh --m ./models/newsqa_seqtok_bertlarge --m_name newsqa_bertlarge --t bert-large-uncased-whole-word-masking --d $DATA --gpuid $gpuid
