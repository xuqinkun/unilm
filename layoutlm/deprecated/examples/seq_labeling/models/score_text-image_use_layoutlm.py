# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import numpy as np
import torch
import sys
import pickle

sys.path.append(os.path.dirname(__file__))

from datasets import load_dataset
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint

import examples.seq_labeling.data.xdoc_perturbation_score as xdoc_perturbation
from examples.seq_labeling.data.data_collator_score import DataCollatorForScore
from examples.seq_labeling.models.modeling_ITA import LayoutlmForImageTextMatching

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


def convert_dataset_to_sents(eval_dataset, overwrite_cache):
    cache_file = Path(training_args.output_dir) / "eval_samples.pickle"
    if cache_file.exists() and not overwrite_cache:
        with cache_file.open('rb') as f:
            sample_index = pickle.load(f)
        return sample_index
    sample_index = {}
    for feature in tqdm(eval_dataset):
        text = "".join(tokenizer.convert_ids_to_tokens(feature['input_ids']))
        sample = {
            "text": text.replace('▁', ''),
            "input_ids": feature['input_ids'],
            "bbox": feature['bbox'],
            "image": feature['image'],
        }
        guid = feature['id']
        group_id = guid[:guid.rindex('-')]
        if group_id not in sample_index:
            sample_index[group_id] = {}
        score = feature['score']
        if score == 1.0:
            key = 'good'
        else:
            key = 'bad'
        if key not in sample_index[group_id]:
            sample_index[group_id][key] = []
        sample_index[group_id][key].append(sample)
    remove_keys = []
    for k, item in sample_index.items():
        if len(item.keys()) != 2:
            remove_keys.append(k)
    for k in remove_keys:
        sample_index.pop(k)
    # torch.save(sample_index, cache_file)
    with cache_file.open('wb') as f:
        pickle.dump(sample_index, f)
    return sample_index


def convert_sample_to_feature(sample: dict):
    text = sample.pop('text')
    feature = tokenizer.pad(sample,
                            padding='max_length',
                            max_length=max_seq_length)
    seq_len = len(feature['bbox'])
    if seq_len < max_seq_length:
        feature['bbox'] = feature['bbox'] + [[0, 0, 0, 0]] * (max_seq_length - seq_len)
    feature = {k: torch.tensor(v[:max_seq_length], device=device).unsqueeze(0) for k, v in feature.items()}
    return text, feature


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = LayoutLMv2Config.from_pretrained(model_args.model_name_or_path,
                                              name_or_path=model_args.model_name_or_path,
                                              num_labels=1)
    max_seq_length = data_args.max_seq_length
    model = LayoutlmForImageTextMatching(config=config, max_seq_length=max_seq_length)
    version = None
    tokenizer = AutoTokenizer.from_pretrained(data_args.encoder_tokenizer, use_fast=True)
    dataset = load_dataset(
        path=Path(xdoc_perturbation.__file__).as_posix(),
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        name=f"x{data_args.doc_type}.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
        max_eval_samples=100,
        max_train_samples=500,
        doc_type=data_args.doc_type,
        cache_dir=data_args.data_cache_dir,
        pred_only=data_args.pred_only,
        is_tar_file=data_args.is_tar_file,
        ocr_path=data_args.ocr_path,
        force_ocr=data_args.force_ocr,
        version=data_args.version,
        output_dir=training_args.output_dir,
        multiple_of_neg_samples=3,
    )
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    num_good_lg_than_bad = 0
    num_eq = 0
    collator_fn = DataCollatorForScore(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )


    def compute_metrics(p):
        """
        l==1 表示good sample，分数应该大于 bad sample
        l==0 表示bad sample
        """
        preds, labels = p
        preds = [p[0] for p in preds]
        num_true = sum([1 if (l == 1 and p > 0.5) or (l == 0 and p < 0.5) else 0 for p, l in zip(preds, labels)])
        return {
            "num_true": num_true,
            "accuracy": num_true / len(labels),
        }


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    local_rank = training_args.local_rank
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    elif local_rank in [0, -1]:
        model_file = Path(last_checkpoint) / "pytorch_model.bin"
        print(f"Load model from file {model_file}")
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)
    if local_rank in [0, -1] and training_args.do_eval:
        # Evaluation
        error_examples = []
        good_examples = []
        eq_samples = []
        max_eval_examples = 500
        eval_samples = convert_dataset_to_sents(eval_dataset, False)
        keys = list(eval_samples.keys())
        eval_indices = np.arange(len(keys))
        np.random.shuffle(eval_indices)
        eval_samples = {keys[i]:eval_samples[keys[i]] for i in eval_indices[:max_eval_examples]}
        device = 'cuda:0'
        model.to(device)
        model.eval()
        groups = []
        for k, item in tqdm(eval_samples.items()):
            good_example = item['good'].pop()
            good_text, x_good = convert_sample_to_feature(good_example)
            good_score = model(**x_good).item()
            pairs = [(good_text, good_score)]
            while len(item['bad']) > 0:
                bad_example = item['bad'].pop()
                bad_text, x_bad = convert_sample_to_feature(bad_example)
                bad_score = model(**x_bad).item()
                pairs.append((bad_text, bad_score))
            pairs.sort(key=lambda k: k[1], reverse=True)
            groups.append((k, pairs))
        # print(f"p(x_good)>p(x_bad): {100 * len(good_examples) / len(eval_samples):.2f}%")
        # print(f"p(x_good)==p(x_bad): {100 * len(eq_samples) / len(eval_samples):.2f}%")
        # for x1, x2, s1, s2 in good_examples[:10]:
        #     print(f"p({x1})={s1:.5f} > p({x2})={s2:.5f}")
        # for x1, x2, s1, s2 in eq_samples[:10]:
        #     print(f"p({x1})={s1:.5f} == p({x2})={s2:.5f}")
        # for x1, x2, s1, s2 in error_examples[:10]:
        #     print(f"p({x1})={s1:.5f} < p({x2})={s2:.5f}")

        for k, group in groups[:10]:
            print(k)
            for item in group:
                print(f"{item[0]}\t{item[1]}")
            print("\n")
