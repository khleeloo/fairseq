#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
import nltk

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import data_utils
from torchmetrics.text.rouge import ROUGEScore
from pprint import pprint
import lmppl
# nltk.download('punkt')  # Download tokenizer models
# nltk.download('averaged_perceptron_tagger')  # Download POS tagger models
# nltk.download('tagsets')  # Download tagset data
# nltk.download('brown')  # Down
# nltk.download('stopwords')
# from nltk.corpus import brown
# from nltk.util import pad_sequence
# from nltk.util import bigrams
# from nltk.util import ngrams
# from nltk.util import everygrams
# from nltk.lm.preprocessing import pad_both_ends
# from nltk.lm.preprocessing import flatten


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        output_perturbation_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-perturbation-{}.txt".format(cfg.dataset.gen_subset),
        )
        output_noise_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-noise-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            f= open(output_perturbation_path, "w", buffering=1, encoding="utf-8")
            p= open(output_noise_path, "w", buffering=1, encoding="utf-8")
            return _main(cfg, h, f,p)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}



def _main(cfg: DictConfig, output_file, output_perturbation_file, output_noise_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)


    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
    
    task.load_hallucination_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)


    
    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=16,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)




    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )
    hall_itr = task.get_batch_iterator(
    dataset=task.hallucination_dataset(cfg.dataset.gen_subset),
    max_tokens=cfg.dataset.max_tokens,
    max_sentences=16,
    max_positions=utils.resolve_max_positions(
        task.max_positions(), *[m.max_positions() for m in models]
    ),
    ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
    seed=cfg.common.seed,
    num_shards=cfg.distributed_training.distributed_world_size,
    shard_id=cfg.distributed_training.distributed_rank,
    num_workers=cfg.dataset.num_workers,
    data_buffer_size=cfg.dataset.data_buffer_size,
).next_epoch_itr(shuffle=False)

    hall_progress = progress_bar.progress_bar(
    hall_itr,
    log_format=cfg.common.log_format,
    log_interval=cfg.common.log_interval,
    default_log_format=( "simple"),
    )

    # hall
    # def generate_perturb_hypos(sample, perturbed_sample_id):
           
    #     # Remove padding
    #     if "src_tokens" in sample["net_input"]:
    #         src_tokens = utils.strip_pad(
    #             sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
    #         )
    #     else:
    #         src_tokens = None

    #     target_tokens = None
    #     if has_target:
    #         target_tokens = (
    #             utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
    #         )


    #     # Either retrieve the original sentences or regenerate them from tokens.
    #     if align_dict is not None:
    #         hall_src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
    #             perturbed_sample_id
    #         )
    #         hall_target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
    #             perturbed_sample_id
    #         )
    #     else:
    #         if src_dict is not None:
    #             hall_src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
    #         else:
    #             hall_src_str = ""
    #         if has_target:
    #             hall_target_str = tgt_dict.string(
    #                 target_tokens,
    #                 cfg.common_eval.post_process,
    #                 escape_unk=True,
    #                 extra_symbols_to_ignore=get_symbols_to_strip_from_output(
    #                     generator
    #                 ),
    #             )

    #     hall_src_str = decode_fn(hall_src_str)
        
    #     hall_target_str=decode_fn(hall_target_str)
        
    
    
    #     # print('\n New_perturbed_sample',perturbed_sample_id)
    #     gen_hall_timer.start()
    #     hall_hypo = task.inference_step(
    #         generator,
    #         models,
    #         sample,
    #         prefix_tokens=prefix_tokens,
    #         constraints=constraints,
    #     )

    #     return hall_hypo , hall_src_str,  align_dict
    


    # Initialize generator
    gen_timer = StopwatchMeter()


    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )


    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)
    cosine_scorer = scoring.build_scorer("cosine", tgt_dict)
    wer_scorer = scoring.build_scorer("wer", tgt_dict)
    rouge_scor = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    # perplexity = lmppl.LM('gpt2')
    perplexity=lmppl.EncoderDecoderLM('google/flan-t5-small')
    perturb_error_count=0
    improvement_counter=0
    wer_score=[]
    bleu_score=[]
    chrf_score=[]
    cosine_score=[]
    cosine_sylable_score=[]
    fluency_score=[]
    # rouge_score=[]
    rouge_precission=[]
    rouge_recall=[]
    rouge_F1=[]
    category=[]
    error_type=[]
    non_perturb_error=0
    perturb_hall_counter=0
    original_hall_counter=0
    wer_count=0
    no_error=0
    perturb_wer_count=0
    oscillation_count=0
    perturb_oscillation_count=0
    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample, hall_sample in zip(progress,hall_progress):

 
                

        sample = utils.move_to_cuda(sample) if use_cuda else sample
        hall_sample  = utils.move_to_cuda(hall_sample ) if use_cuda else hall_sample 
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hypos = task.inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        # print('\n New_perturbed_sample',perturbed_sample_id)
    
        hall_hypos = task.inference_step(
            generator,
            models,
            hall_sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
    # num_generated_hall_tokens = sum(len(h[0]["tokens"]) for h in hall_hypos)
    # gen_hall_timer.stop(num_generated_hall_tokens)
    
        for i, (sample_id,hall_sample_id) in enumerate(zip(sample["id"].tolist(),hall_sample["id"].tolist())):
            if sample_id!=hall_sample_id:
                print(sample_id, hall_sample_id)
                print('Dataset alignment error!')
                break
        

            def processing_hypos(sample,sample_id):
                has_target = sample["target"] is not None
                if "src_tokens" in sample["net_input"]:
                    src_tokens = utils.strip_pad(
                        sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                    )
                else:
                    src_tokens = None

                target_tokens = None
                if has_target:
                    target_tokens = (
                        utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                    )

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(cfg.dataset.gen_subset).src.get_original_text(
                        sample_id
                    )
                    target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                        sample_id
                    )
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(
                            target_tokens,
                            cfg.common_eval.post_process,
                            escape_unk=True,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                                generator
                            ),
                        )

                src_str = decode_fn(src_str)
                if has_target:
                    target_str = decode_fn(target_str)


                if not cfg.common_eval.quiet:
                    if src_dict is not None:
                        print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                    if has_target:
                        print("T-{}\t{}".format(sample_id, target_str), file=output_file)

                return src_str,src_dict, target_tokens,target_str,align_dict

            src_str,src_dict, target_tokens,target_str,align_dict=processing_hypos(sample, sample_id)
        
 
            # Process top predictions
            for j, (hypo, perturb_hypo) in enumerate(zip(hypos[i][: cfg.generation.nbest],hall_hypos[i][:cfg.generation.nbest])):



                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
    
                
                detok_hypo_str = decode_fn(hypo_str)




                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [",".join(src_probs) for src_probs in alignment]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )

                # Score only the top hypothesis
                if has_target and j==0 :
                    if (
                        align_dict is not None
                        or cfg.common_eval.post_process is not None
                    ):
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True
                        )
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True
                        )

                    if hasattr(scorer, "add_string"):
                        scorer.add_string(target_str, detok_hypo_str)
                        wer_scorer.reset()
                        wer_scorer.add_string(target_str, detok_hypo_str)
                        wsc=wer_scorer.score()
                        # if 'wer' in cfg.common_eval.results_path:                      
                        #     d={'WER':wer_score}
                        #     df=pd.DataFrame(data=d)
                        #     # df.to_csv('{}/perturb1_wer_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)

                        # elif 'bleu' in cfg.common_eval.results_path:  
                        #     d={'BLEU':bleu_score}
                        #     df=pd.DataFrame(data=d)
                        #     df.to_csv('{}/perturb_bleu_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)
                        # elif 'hall' in cfg.common_eval.results_path:  
                        #     d={'cosine':cosine_score,'cosine_sylable' :cosine_sylable_score, 'perplexity':fluency_score, 'rouge precission':rouge_precission,'rouge recall':rouge_recall,'rouge F1':rouge_F1}
                        #     df=pd.DataFrame(data=d)
                        #     df.to_csv('{}/hall_metric_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)
                        # wer_score.append(wsc)
                        
                        # d={'WER':wer_score}
                        # df=pd.DataFrame(data=d)
            
                        # df.to_csv('{}/sentence_wer_{}.tsv'.format(cfg.dataset.gen_subset) ,sep='\t',index=False)

                        
                    # else:
                    #     scorer.reset(one_init=True)
                    #     scorer.add(target_tokens, hypo_tokens)

                        
                    #     original_score=scorer.score(2)
          

                        
                        
                        oscillation=''
                        phonetic_error=''
                        hallucination=''
                        cat=''
                        error=''
                        """hallucination detection algorithm"""
                        if wsc <=30:
                            
                            assert sample_id==hall_sample_id 
          
                            hall_src_str,src_dict, target_tokens,hall_target_str,align_dict=processing_hypos(hall_sample, hall_sample_id)
                            perturb_hypo_tokens, perturb_hypo_str, _ = utils.post_process_prediction(
                            hypo_tokens=perturb_hypo["tokens"].int().cpu(),
                            src_str=hall_src_str,
                            alignment=perturb_hypo["alignment"],
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=cfg.common_eval.post_process,
                            extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                        )
                            perturb_detok_hypo_str=decode_fn(perturb_hypo_str)
         
                            wer_scorer.add_string(target_str, perturb_detok_hypo_str)
                            ptb=wer_scorer.score()
                            
                            wer_score.append(ptb)
                            # scorer.reset(one_init=True)
                            # scorer.add(target_tokens,perturb_hypo_tokens)
                            # perturbed_score=scorer.score(2)

                            perplexity_perturb = perplexity.get_perplexity(target_str,perturb_detok_hypo_str)
                            cosine_scorer.add_string(target_str, perturb_detok_hypo_str)
                            cosine=cosine_scorer.score("word")
                            cosine_syllable=cosine_scorer.score("syllable")
                            fluency_score.append(perplexity_perturb)
                            cosine_score.append(cosine)
                            cosine_sylable_score.append(cosine_syllable)
                            rouge=rouge_scor.score(target_str, perturb_detok_hypo_str)
                            rouge_precission.append(rouge['rouge1'][0])
                            rouge_recall.append(rouge['rouge1'][1])
                            rouge_F1.append(rouge['rouge1'][2])
                                            
                            if ptb>30:
                                cat='perturbed'
                                  
                                perturb_error_count+=1
                                             
                                if cosine<=0.2 and perplexity_perturb <=200:
                                    perturb_hall_counter+=1
                                    error='hallucination'
                                    hallucination=perturb_detok_hypo_str
                                      
                                elif cosine>0.2 and perplexity_perturb <=200:
                                    error='phonetic error'
                                    perturb_wer_count+=1
                                    phonetic_error=perturb_detok_hypo_str

                                elif cosine>0.2 and perplexity_perturb >200:
                                    error='oscillation'                            
                                    oscillation_count+=1
                                    oscillation=perturb_detok_hypo_str

                                else:
                                    error='oscillation'
                                    oscillation=perturb_detok_hypo_str
                                    oscillation_count+=1
                                
                            else:
                                error='no error'
                                no_error+=1
                                cat='not perturbed'

                            category.append(cat)
                            error_type.append(error)
                            if hallucination  or phonetic_error  or oscillation :
                                print( " \n Sample ID {} \n  ORIGINAL SCORE:{} ,  PERTURBED SCORE: {} \n Pertrub WER COUNT:{} PHONETIC ERROR:{} \n Pertrub HALLUCINATION COUNT:{} HALLUCINATION: {}\n  Pertrub OSCILLATION COUNT:{} OSCILLATION: {}\nTarget Str:{} \n Hypo str: {} \n error counter {} \n no error counter {}".format(
                                    sample_id,wsc,ptb,perturb_wer_count,phonetic_error,perturb_hall_counter, hallucination, oscillation_count, oscillation,target_str, detok_hypo_str , perturb_error_count,no_error
                                    ),
                                    file=output_perturbation_file,
                                    )   
            
                        else: 
                            cat='not perturbed'
                            wer_score.append(wsc) 
                            non_perturb_error+=1
                            
                            cosine_scorer.add_string(target_str, detok_hypo_str)
                            cosine=cosine_scorer.score("word")
                            cosine_syllable=cosine_scorer.score("syllable")
                            perplexity_original = perplexity.get_perplexity(target_str,detok_hypo_str)
                            fluency_score.append(perplexity_original )
                            cosine_score.append(cosine)
                            cosine_sylable_score.append(cosine_syllable)
                            rouge=rouge_scor.score(target_str, detok_hypo_str)
                         
                            rouge_precission.append(rouge['rouge1'][0])
                            rouge_recall.append(rouge['rouge1'][1])
                            rouge_F1.append(rouge['rouge1'][2])

                           
                            if  cosine<=0.2 and perplexity_original <=200:
                                original_hall_counter+=1
                                hallucination=detok_hypo_str
                                error='hallucination'
                   
                            elif cosine>0.2 and perplexity_original <=200:
                                wer_count+=1
                                phonetic_error=detok_hypo_str
                                error='phonetic error'
                            else:
                                error='oscillation'
                                oscillation=detok_hypo_str
                                oscillation_count+=1
                            category.append(cat)
                            error_type.append(error)
                            if hallucination  or phonetic_error  or oscillation :
                                print( " \n Sample ID {} \n  ORIGINAL SCORE:{}    \n Original WER COUNT:{}  PHONETIC ERROR: {} \n Original OSCILLATION COUNT:{}   OSCILLATION: {}  \n Original HALLUCINATION COUNT:{} HALLUCINATION: {}\n Target Str:{} \n Error counter{}".format(
                                    sample_id,wsc,wer_count,phonetic_error, oscillation_count, oscillation, original_hall_counter,hallucination,target_str, non_perturb_error
                                    ),
            
                                    file=output_noise_file,
                                )    
                                        
        if 'wer' in cfg.common_eval.results_path:                      
            
            d={'WER':wer_score}
            df=pd.DataFrame(data=d)
            df.to_csv('{}/perturb_wer_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)

        elif 'bleu' in cfg.common_eval.results_path:  
            d={'BLEU':bleu_score}
            df=pd.DataFrame(data=d)
            df.to_csv('{}/perturb_bleu_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)
        elif 'hall' in cfg.common_eval.results_path:  
            d={'cosine':cosine_score,'cosine_sylable' :cosine_sylable_score, 'perplexity':fluency_score, 'rouge precission':rouge_precission,'rouge recall':rouge_recall,'rouge F1':rouge_F1,'if perturbed':category,'error type': error_type}
            df=pd.DataFrame(data=d)
            df.to_csv('{}/perturb_hall_metric_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)
            w={'WER':wer_score}
            wf=pd.DataFrame(data=w)
            wf.to_csv('{}/perturb_wer_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)
        else:
            d={'ChRF':chrf_score}
            df=pd.DataFrame(data=d)
            df.to_csv('{}/perturb_chrf_{}.tsv'.format(cfg.common_eval.results_path, cfg.dataset.gen_subset) ,sep='\t',index=False)

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})
        num_sentences += (
            sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
        )

    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {:,} sentences ({:,} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            num_sentences / gen_timer.sum,
            1.0 / gen_timer.avg,
        )
    )
    if has_target:
        if cfg.bpe and not cfg.generation.sacrebleu:
            if cfg.common_eval.post_process:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            "Generate {} with beam={}: {}".format(
                cfg.dataset.gen_subset, cfg.generation.beam, scorer.result_string()
            ),
            file=output_file,
        )

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
