#!/bin/bash

LS_ROOT=/home/rmfrieske/datasets/
FAIRSEQ=/home/rmfrieske/fairseq/
TENSOR_LOG=/home/rmfrieske/tensor_log/

MODEL=UR
mkdir /home/rmfrieske/checkpoints/${MODEL}
SAVE_DIR=/home/rmfrieske/checkpoints/${MODEL}/
export PYTHONPATH='/home/rmfrieske/fairseq/'


python ${FAIRSEQ}fairseq_cli/train.py ${LS_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train-UR--valid-subset dev-clean,dev-other \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed  \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8 --tensorboard-logdir ${TENSOR_LOG}${MODEL} 



if ! avg_last_10_checkpoint.pt; then
	CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
	python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  	--num-epoch-checkpoints 10 \
  	--output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
fi

for SUBSET in dev-clean dev-other test-clean test-other; do
 python ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring wer --results-path ${SAVE_DIR}wer   
done

for SUBSET in dev-clean dev-other test-clean test-other; do
  python ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring bleu --results-path ${SAVE_DIR}bleu
done

for SUBSET in dev-clean dev-other test-clean test-other; do
 python  ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring chrf --results-path ${SAVE_DIR}chrf
done


