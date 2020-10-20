#!/bin/bash

set -e

m_name=${m_name}
d=${d}


if [ "$1" == "-h" ]; then
  echo "Aggregate scores from model predictions."
  echo "   --m_name      A brief name of the QA model, used to compose output path, e.g., in {robertabase, robertalarge, bertbase, bertlarge, distilbert}"
  echo "   --d           A list of dataset types, separated by comma, must be in {gender, country, religion, ethnicity}"
  echo "   --extra       A filler; specify this if to use source file generated with extra filler, e.g., newsqa"
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

echo "======================================="
echo "       AGGREGATING PREDICTIONS"
echo "======================================="


m_name=$(echo $m_name | tr "," "\n")
echo ">> Models to process: "$m_name

d=$(echo $d | tr "," "\n")
echo ">> Datasets to process: "$d

for m_namei in $m_name
do
	for di in $d
	do
		# TODO, should support group by subj as well
		case "$di" in
	    	gender)
	    	    GROUP_BY="gender_act"
	    	    ;;
	    	*)
	    	    GROUP_BY="subj_act"
	    	    ;;
		esac

		if [[ -n $extra ]]; then
    	  FILE=${extra}_${m_namei}_${di}
    	else
    	  FILE=${m_namei}_${di}
    	fi

    	echo ">> Input file "${FILE}.output.json
	
		python3 analysis.py \
	    	--metrics subj_bias,pos_err,attr_err \
	    	--input ./data/${FILE}.output.json \
	    	--group_by ${GROUP_BY} --verbose 1 | tee ./data/${m_namei}_${di}.log.txt
	done
done

exit 0