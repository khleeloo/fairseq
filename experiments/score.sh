#!/bin/bash


FAIRSEQ=/home/rmfrieske/git/fairseq/fairseq_cli/
REF =
SYS 

python  ${FAIRSEQ}score.py [-h] -s $SYS -r $REF -o 2 
                     --sentence-bleu