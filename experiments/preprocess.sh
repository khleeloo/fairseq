#!/bin/bash

SCRIPT_PATH=/home/rmfrieske/git/fairseq/examples/speech_to_text
RELATIVE_PATH=examples/speech_to_text
LS_ROOT='/home/rmfrieske/git/datasets/'

export PYTHONPATH='/home/rmfrieske/git/fairseq/'

python $SCRIPT_PATH/prep_librispeech_data.py \
  --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000