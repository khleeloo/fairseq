#!/bin/bash

MODEL=UU
EXPERIMENT='baseline'

FAIRSEQ=/home/rmfrieske/fairseq/fairseq_cli/

blue_cutoff=10

SAVE_DIR=/home/rmfrieske/checkpoints/${MODEL}/${EXPERIMENT}/bleu/
export PYTHONPATH='/home/rmfrieske/fairseq/'



REF='target-'
HYP='hypotheses-'
GEN='generate-'

for SUBSET in dev-clean dev-other test-clean test-other ; do
rm ${SAVE_DIR}example_hallucinations_$SUBSET.txt 
rm ${SAVE_DIR}/hallucination-stats-${SUBSET}.txt 

python  ${FAIRSEQ}score.py -s $SAVE_DIR$HYP$SUBSET.txt \
-r $SAVE_DIR$REF$SUBSET.txt -o 2 --sentence-bleu  >  ${SAVE_DIR}'sentencescore-'${SUBSET}'.txt'


count=0
for j in `grep BLEU2   /home/rmfrieske/checkpoints/${MODEL}/baseline/bleu/sentencescore-${SUBSET}.txt | cut -d " " -f1,4`;do
          
            if [[ $j =~ [0-9]+$ ]]; then
                i=$j          
            fi
   
    printf -v int '%d\n' $j 2>/dev/null
    if [ ${int} -lt "${blue_cutoff}" ] ;then
        if [ -n "$i" ]; then
            printf '%s\n' $(($i+1)) >>${SAVE_DIR}/tmp_$SUBSET.txt
            
        fi
    let  count++
    fi
done

 all=$(wc -l <  ${SAVE_DIR}/sentencescore-${SUBSET}.txt )

 echo -e "${SUBSET} **HALLUCINATION STATS** ${EXPERIMENT}\n Blue cutoff value: $blue_cutoff \n Dataset size: $all\n Overall number of hallucinations: $count \n Hallucination ratio: " >> ${SAVE_DIR}/hallucination-stats-${SUBSET}.txt 
 printf "%f\n" $((10**6 * $count/$all * 100))e-6 >> ${SAVE_DIR}/hallucination-stats-${SUBSET}.txt 
 

sed -i '1d;/^$/d'  ${SAVE_DIR}tmp_$SUBSET.txt

sed -n "$(sed 's/$/p/' ${SAVE_DIR}tmp_$SUBSET.txt )" $SAVE_DIR$HYP$SUBSET.txt  > tmp$HYP$SUBSET
sed -n "$(sed 's/$/p/' ${SAVE_DIR}tmp_$SUBSET.txt )" $SAVE_DIR$REF$SUBSET.txt  > tmp$REF$SUBSET
rm  ${SAVE_DIR}tmp_$SUBSET.txt 
# rm ${SAVE_DIR}/tmp_$SUBSET 

# open input files
exec {fdA}<"tmp$REF$SUBSET"
exec {fdB}<"tmp$HYP$SUBSET"

while read -r -u "$fdA" lineA && read -r -u "$fdB" lineB
do
    echo "Reference: $lineA" >> ${SAVE_DIR}example_hallucinations_$SUBSET.txt 
    echo "Hypothesis: $lineB" >>${SAVE_DIR}example_hallucinations_$SUBSET.txt 
done

exec {fdA}>&- {fdB}>&- # close input files

rm "tmp$HYP$SUBSET" "tmp$REF$SUBSET"

done