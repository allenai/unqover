#/bin/bash

set -e

echo "======================================="
echo    "      GENERATING QUESTIONS"
echo "======================================="

echo ">> Will generate 6GB of data(~13.5M questions)"
mkdir -p ./data
echo "========  GENDER ========="
TYPE=slot_act_map
SUBJ=mixed_gender
SLOT=gender_noact
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
python3 -m templates.generate_underspecified_templates --template_type ${TYPE} \
      --subj $SUBJ --act $ACT --slot $SLOT \
        --output ./data/${FILE}.source.json

TYPE=slot_act_map
SUBJ=mixed_gender_roberta
SLOT=gender_noact_lm
ACT=occupation_rev1
FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
python3 -m templates.generate_underspecified_templates --template_type $TYPE \
      --subj $SUBJ --act $ACT --slot $SLOT \
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
    SUBJ=${CLASS}_roberta
    SLOT=${CLASS}_noact_lm
    ACT=biased_${CLASS}
    FILE=slotmap_${SUBJ//_}_${ACT//_}_${SLOT//_}
    python3 -m templates.generate_underspecified_templates --template_type $TYPE \
          --subj $SUBJ --act $ACT --slot $SLOT \
            --output ./data/${FILE}.source.json
done

