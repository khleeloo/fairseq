#!/bin/bash

MODEL=UU
EXPERIMENT='white_noise_first'

FAIRSEQ=/home/rmfrieske/fairseq/fairseq_cli/



SAVE_DIR=/home/rmfrieske/checkpoints/${MODEL}/${EXPERIMENT}/bleu/
export PYTHONPATH='/home/rmfrieske/fairseq/'

REF='target-'
HYP='hypotheses-'
GEN='generate-'

for SUBSET in  test-clean ; do
python  ${FAIRSEQ}score.py -s $SAVE_DIR$HYP$SUBSET.txt \
-r $SAVE_DIR$REF$SUBSET.txt -o 2 --sentence-bleu  >  ${SAVE_DIR}$SCORE/sentencescore-${SUBSET}.txt

                   
done