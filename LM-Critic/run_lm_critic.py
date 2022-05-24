# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import layoutlm.deprecated.examples.seq_labeling.data.xdoc_perturbation_score as xdoc_perturbation
from critic.critic import gpt2_critic

if __name__ == '__main__':
    version = None
    tokenizer = AutoTokenizer.from_pretrained("/home/std2020/xuqinkun/model/xlm-roberta-base", use_fast=True)

    dataset = load_dataset(
        path=Path(xdoc_perturbation.__file__).as_posix(),
        data_dir="/home/std2020/xuqinkun/data/doc_image/contract",
        tokenizer=tokenizer,
        name='xcontract.en',
        additional_langs=None,
        keep_in_memory=True,
        doc_type='contract',
        cache_dir=None,
        pred_only=False,
        is_tar_file=True,
        version='0.0.2',
        output_dir='/home/std2020/xuqinkun/model/contract_ITA',
    )
    eval_dataset = dataset['validation']
    nb_gt = 0
    error_examples = []
    good_examples = []
    device = 'cuda:0'
    for feature in tqdm(eval_dataset):

        image = torch.tensor(feature['image'], dtype=torch.float32, device=device).unsqueeze(0)
        good_inputs = feature['good_inputs']
        bad_inputs = feature['bad_inputs']
        if len(good_inputs) == 0 or len(bad_inputs) == 0:
            continue

        good_example = tokenizer.convert_ids_to_tokens(good_inputs)
        bad_example = tokenizer.convert_ids_to_tokens(bad_inputs)
        good_example = "".join(good_example).replace('▁', '')
        bad_example = "".join(bad_example).replace('▁', '')
        good_ret = gpt2_critic(good_example, verbose=0)
        bad_ret = gpt2_critic(bad_example, verbose=0)
        if good_ret is None or bad_ret is None:
            continue
        good_score = good_ret[1]
        bad_score = bad_ret[1]
        if good_score > bad_score:
            nb_gt += 1
            good_examples.append((good_example, bad_example))
        else:
            error_examples.append((good_example, bad_example))

    print(f"p(x_good)>p(x_bad): {nb_gt / len(eval_dataset)}\n")

    print("\nGood \tBad\n")
    for x1, x2 in good_examples[:10]:
        print(f"{x1}\t{x2}\n")
    print("\nGood \tBad\n")

    for x1, x2 in error_examples[:10]:
        print(f"{x1}\t{x2}\n")