#/bin/bash

set -e

#echo "======================================="
#echo    "      INSTALLING DEPENDENCIES"
#echo "======================================="
#
#echo -e "Already installed? [Y/*]\c"
#read dep
#if [[ "$dep" == "Y" ]]; then
#    echo "Skipped!"
#else
#    pip install -r requirements.txt
#fi

echo "======================================="
echo    "      DOWNLOADING DATA"
echo "======================================="

#echo "How do you plan to use the code?"
#echo "1) I just want to generate the output scores"
#echo "   >> This will download 6GB of predictions that expands to ~60GB of data)"
#echo "2) [TODO] I will produce the predictions using your models"
#echo "   >> This will download only the models but will take hours to generate predictions"
#echo "3) I want to produce the predictions only for my models"
#echo "   >> No need to download any data"

#echo -e "Choice: \c"
#read choice

#case $choice in
#    1)
#        wget https://ai2-datasets.s3.amazonaws.com/unqover_model_dumps/data.zip
#        ;;
#    2)
#        wget https://ai2-datasets.s3.amazonaws.com/unqover_model_dumps/models.zip
#        ;;
#    *)
#        ;;
#
#esac

wget https://ai2-datasets.s3.amazonaws.com/unqover_model_dumps/data.zip

unzip data.zip

echo "Removing data.zip"
rm data.zip

echo "Downloading Complete!"
