#!/bin/bash

set -e

m=${m}
m_name=${m_name}
d=${d}
gpuid=${gpuid:-0}
extra=${extra} # default to empty meaning no filler to use

if [ "$1" == "-h" ]; then
  echo "Generate predictions from QA models saved in HuggingFace's format."
  echo "   --m           Path to the directory that contains HF model"
  echo "   --m_name      A brief name of the QA model, used to compose output path, must be in {robertabase, robertalarge, bertbase, bertlarge, distilbert}, optionally with 'newsqa_' prefix"
  echo "   --d           A list of dataset types, separated by comma, must be in {gender, country, religion, ethnicity}"
  echo "   --gpuid       The GPU device index to use, default to 0"
  echo "   --extra       A filler; specify this if to use source file generated with extra filler, e.g., newsqa"
  echo "   -h            Print the help message and exit"
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

echo "======================================="
echo "        PREDICTING QUESTIONS"
echo "======================================="


d=$(echo $d | tr "," "\n")
echo ">> Datasets to process: "$d

for di in $d
do
    echo "======================================="
    echo ">> Running QA model (HuggingFace) "${m}" on "${di}" data"
    echo ">> Will dump predictions to ./data/"${m_name}_${di}.output.json

    TYPE=slot_act_map
    case "${di}" in
        "gender")
            SUBJ=mixed_gender
            SLOT=gender_noact
            ACT=occupation_rev1
           ;;
        "country" | "religion" | "ethnicity")
            SUBJ=${di}
            SLOT=${di}_noact
            ACT=biased_${di}
            ;;
        *)
            echo "Can not handle subject class: ${di}"
            exit 1
            ;;
    esac

    if [[ -n $extra ]]; then
      FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}_${extra}
    else
      FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    fi

    echo ">> Input file "${FILE}

    python3 -u -m qa_hf.predict --gpuid ${gpuid} \
    --hf_model tli8hf/${m} \
    --input ${FILE}.source.json --output ./data/${m_name}_${di}.output.json
done


exit 0