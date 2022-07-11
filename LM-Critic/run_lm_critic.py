# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
import layoutlm.deprecated.examples.seq_labeling.data.xdoc_perturbation_score as xdoc_perturbation
from critic.critic import gpt2_critic


def convert_dataset_to_sents(eval_dataset):
    eval_samples = []
    for feature in eval_dataset:
        good_inputs = feature['good_inputs']
        bad_inputs = feature['bad_inputs']
        if len(good_inputs) == 0 or len(bad_inputs) == 0 or good_inputs == bad_inputs:
            continue
        good_example = tokenizer.convert_ids_to_tokens(good_inputs)
        bad_example = tokenizer.convert_ids_to_tokens(bad_inputs)
        good_example = "".join(good_example).replace('▁', '')
        bad_example = "".join(bad_example).replace('▁', '')
        if good_example == bad_example:
            continue
        eval_samples.append((good_example, bad_example))
    return eval_samples


if __name__ == '__main__':
    version = None
    tokenizer = AutoTokenizer.from_pretrained("/home/std2020/xuqinkun/model/xlm-roberta-base", use_fast=True)
    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dataset = load_dataset(
        path=Path(xdoc_perturbation.__file__).as_posix(),
        data_dir="/home/std2020/xuqinkun/data/doc_image/contract",
        tokenizer=tokenizer,
        name='xcontract.zh',
        additional_langs=None,
        keep_in_memory=True,
        doc_type='contract',
        cache_dir=None,
        pred_only=False,
        is_tar_file=True,
        version=data_args.version,
        output_dir='/home/std2020/xuqinkun/model/contract_ITA',
    )
    eval_dataset = dataset['validation']
    error_examples = []
    good_examples = []
    eq_examples = []
    device = 'cuda:0'
    max_eval_examples = 500
    eval_samples = convert_dataset_to_sents(eval_dataset)
    eval_indices = np.arange(len(eval_samples))
    np.random.shuffle(eval_indices)

    eval_samples = [eval_samples[i] for i in eval_indices[:max_eval_examples]]
    for x_good, x_bad in tqdm(eval_samples):
        good_ret = gpt2_critic(x_good, verbose=0)
        bad_ret = gpt2_critic(x_bad, verbose=0)
        if good_ret is None or bad_ret is None:
            continue
        good_score = good_ret[1]
        bad_score = bad_ret[1]
        if good_score > bad_score:
            good_examples.append((x_good, x_bad))
        elif good_score < bad_score:
            error_examples.append((x_good, x_bad))
        else:
            eq_examples.append((x_good, x_bad))

    print(f"p(x_good)>p(x_bad): {100*len(good_examples) / len(eval_samples):.2f}%\n")
    for x1, x2 in good_examples[:10]:
        print(f"p({x1}) > p({x2})")
    for x1, x2 in error_examples[:10]:
        print(f"p({x1})<p({x2})")
    for x1, x2 in eq_examples[:10]:
        print(f"p({x1})==p({x2})")
