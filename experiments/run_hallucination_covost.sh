#!/bin/bash
EXPERIMENT='baseline'
MODEL=covost

LS_ROOT='/home/rmfrieske/datasets/covost/en/'
# LS_ROOT='/home/rmfrieske/datasets/'
HALL_ROOT='/home/rmfrieske/datasets/perturbed_2/'

mkdir /home/rmfrieske/checkpoints/${MODEL}
mkdir /home/rmfrieske/checkpoints/${MODEL}/${EXPERIMENT}/
SAVE_DIR=/home/rmfrieske/checkpoints/${MODEL}/${EXPERIMENT}/
CHECKPOINT_DIR=/home/rmfrieske/checkpoints/${MODEL}/
FAIRSEQ=/home/rmfrieske/fairseq/
TENSOR_LOG=/home/rmfrieske/tensor_log/${MODEL}/

export PYTHONPATH='/home/rmfrieske/fairseq/'

python ${FAIRSEQ}fairseq_cli/train.py ${LS_ROOT} \
  --config-yaml config.yaml --train-subset train_covost --valid-subset dev_covost \
  --save-dir ${SAVE_DIR} --num-workers 4 --max-tokens 50000 --max-update 60000 --restore-file ${CHECKPOINT_DIR}'covost2_en_asr_transformer_s.pt' \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --report-accuracy --arch s2t_transformer_s --dropout 0.15 --reset-optimizer --optimizer adam --lr 2e-3 \
  --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
   --tensorboard-logdir ${TENSOR_LOG}

# python ${FAIRSEQ}fairseq_cli/train.py ${LS_ROOT} \
#   --config-yaml config.yaml --train-subset train_covost --valid-subset dev_covost \
#   --save-dir ${SAVE_DIR} --num-workers 4 --max-tokens 50000 --max-update 60000  \
#   --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#   --report-accuracy --arch s2t_transformer_s --dropout 0.15  --optimizer adam --lr 2e-3 \
#   --lr-scheduler inverse_sqrt --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
#    --tensorboard-logdir ${TENSOR_LOG}

# CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
# # CHECKPOINT_FILENAME=librispeech_transformer_l.pt
# if ! avg_last_10_checkpoint.pt; then
# 	CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
# 	python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
#   	--num-epoch-checkpoints 10 \
#   	--output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
# fi

# SCORE=bleu
# for SUBSET in dev-clean dev-other test-clean test-other ; do
# python  ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT}  --config-yaml config.yaml --gen-subset ${SUBSET} \
#     --task speech_to_text --arch s2t_transformer_s --path ${CHECKPOINT_DIR}/${CHECKPOINT_FILENAME} \
#     --max-tokens 50000 --beam 5 --scoring $SCORE --results-path ${SAVE_DIR}$SCORE 

# grep ^T ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f2- > ${SAVE_DIR}$SCORE/target-${SUBSET}.txt
# grep ^D ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f3- > ${SAVE_DIR}$SCORE/hypotheses-${SUBSET}.txt
# done


# SCORE=chrf
# for SUBSET in dev-clean dev-other test-clean test-other ; do
# python  ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT}  --config-yaml config.yaml --gen-subset ${SUBSET} \
#     --task speech_to_text --arch s2t_transformer_s --path ${CHECKPOINT_DIR}/${CHECKPOINT_FILENAME} \
#     --max-tokens 50000 --beam 5 --scoring $SCORE --results-path ${SAVE_DIR}$SCORE 

# grep ^T ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f2- > ${SAVE_DIR}$SCORE/target-${SUBSET}.txt
# grep ^D ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f3- > ${SAVE_DIR}$SCORE/hypotheses-${SUBSET}.txt
# done



# SCORE=wer
# for SUBSET in dev-clean dev-other test-clean test-other ; do
# python  ${FAIRSEQ}fairseq_cli/generate.py ${LS_ROOT}  --config-yaml config.yaml --gen-subset ${SUBSET} \
#     --task speech_to_text --arch s2t_transformer_s --path ${CHECKPOINT_DIR}/${CHECKPOINT_FILENAME} \
#     --max-tokens 50000 --beam 5 --scoring $SCORE --results-path ${SAVE_DIR}$SCORE 

# grep ^T ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f2- > ${SAVE_DIR}$SCORE/target-${SUBSET}.txt
# grep ^D ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f3- > ${SAVE_DIR}$SCORE/hypotheses-${SUBSET}.txt
# done

# #hall

# SCORE=hall
# for SUBSET in dev_covost test_covost; do
#  python  ${FAIRSEQ}fairseq_cli/generate_hallucination.py ${LS_ROOT} ${HALL_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET}  \
#     --task speech_to_text_hallucination --path ${CHECKPOINT_DIR}/${CHECKPOINT_FILENAME}  --arch s2t_hallucination_transformer_s   \
#     --max-tokens 50000 --beam 5 --scoring 'wer' --results-path ${SAVE_DIR}$SCORE 

# grep ^T ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f2- > ${SAVE_DIR}$SCORE/target-${SUBSET}.txt
# grep ^D ${SAVE_DIR}$SCORE/generate-${SUBSET}.txt | cut -f3- > ${SAVE_DIR}$SCORE/hypotheses-${SUBSET}.txt
# done