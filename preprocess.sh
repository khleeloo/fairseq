#!/bin/bash

SCRIPT_PATH=/home/rmfrieske/fairseq/examples/speech_to_text
# RELATIVE_PATH=examples/speech_to_text
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> aabf10b4f84b1d406938739c1746c85ce2f7d9b0
LS_ROOT='/home/rmfrieske/datasets/covost/en/'
# LS_ROOT='/home/rmfrieske/datasets'

export PYTHONPATH='/home/rmfrieske/fairseq/'

python $SCRIPT_PATH/prep_covost_data_english.py \
  --data-root ${LS_ROOT} --vocab-type char 

# python $SCRIPT_PATH/prep_librispeech_data.py \
# --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
<<<<<<< HEAD
=======
=======
# LS_ROOT='/home/rmfrieske/datasets/covost/'
LS_ROOT='/home/rmfrieske/datasets'

export PYTHONPATH='/home/rmfrieske/fairseq/'

# python $SCRIPT_PATH/prep_covost_data_english.py \
#   --data-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000

python $SCRIPT_PATH/prep_librispeech_data.py \
--output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
>>>>>>> cd8858ae8b4ee4bd0c93ceac5cb5d340a95017ac
>>>>>>> aabf10b4f84b1d406938739c1746c85ce2f7d9b0

