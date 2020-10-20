#/bin/bash

set -e


if [ "$1" == "-h" ]; then
  echo "Generate underspecified questions for all models and bias classes."
  echo "   -h           Print the help message and exit"
  exit 0
fi

echo "======================================="
echo "         GENERATING QUESTIONS"
echo "======================================="

echo ">> Will generate 6GB of data(~13.5M questions)"
mkdir -p ./data
echo "========  GENDER ========="

# for squad (by default)
TYPE=slot_act_map
SUBJ=mixed_gender
SLOT=gender_noact
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
      --subj $SUBJ --act $ACT --slot $SLOT \
      --output ./data/${FILE}.source.json

# for newsqa (with --filler)
TYPE=slot_act_map
SUBJ=mixed_gender
SLOT=gender_noact
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}_newsqa
python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
      --subj $SUBJ --act $ACT --slot $SLOT --filler newsqa \
      --output ./data/${FILE}.source.json

TYPE=slot_act_map
SUBJ=mixed_gender_roberta
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
      --subj $SUBJ --act $ACT --slot $SLOT --lm_mask "<mask>" \
      --output ./data/${FILE}.source.json


TYPE=slot_act_map
SUBJ=mixed_gender_bert
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
      --subj $SUBJ --act $ACT --slot $SLOT --lm_mask "[MASK]" \
        --output ./data/${FILE}.source.json


for CLASS in ethnicity country religion
do
    echo "========  ${CLASS^^} ========="

    TYPE=slot_act_map
    SUBJ=$CLASS
    SLOT=${CLASS}_noact
    ACT=biased_${CLASS}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
          --subj $SUBJ --act $ACT --slot $SLOT \
          --output ./data/${FILE}.source.json

    TYPE=slot_act_map
    SUBJ=$CLASS
    SLOT=${CLASS}_noact
    ACT=biased_${CLASS}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}_newsqa
    python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
          --subj $SUBJ --act $ACT --slot $SLOT --filler newsqa \
          --output ./data/${FILE}.source.json

    TYPE=slot_act_map
    SUBJ=${CLASS}_roberta
    SLOT=${CLASS}_noact_lm
    ACT=biased_${CLASS}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
          --subj $SUBJ --act $ACT --slot $SLOT --lm_mask "<mask>" \
          --output ./data/${FILE}.source.json

    TYPE=slot_act_map
    SUBJ=${CLASS}_bert
    SLOT=${CLASS}_noact_lm
    ACT=biased_${CLASS}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
          --subj $SUBJ --act $ACT --slot $SLOT --lm_mask "[MASK]" \
          --output ./data/${FILE}.source.json
done

exit 0
