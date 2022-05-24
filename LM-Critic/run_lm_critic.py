# -*- coding: utf-8 -*-
import torch
import numpy as np
from pathlib import Path
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
        name='xcontract.zh',
        additional_langs=None,
        keep_in_memory=True,
        doc_type='contract',
        cache_dir=None,
        pred_only=False,
        is_tar_file=True,
        version='0.0.3',
        output_dir='/home/std2020/xuqinkun/model/contract_ITA',
    )
    eval_dataset = dataset['validation']
    error_examples = []
    good_examples = []
    eq_examples = []
    device = 'cuda:0'
    max_eval_examples = 500
    eval_indices = np.arange(len(eval_dataset))
    np.random.shuffle(eval_indices)
    eval_samples = eval_dataset.select(eval_indices[:max_eval_examples])
    for feature in tqdm(eval_samples):

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
            good_examples.append((good_example, bad_example))
        elif good_score < bad_score:
            error_examples.append((good_example, bad_example))
        else:
            eq_examples.append((good_example, bad_example))

    print(f"p(x_good)>p(x_bad): {100*len(good_examples) / len(eval_samples):.2f}%\n")
    print(f"p(x_good)==p(x_bad): {100*len(eq_examples) / len(eval_samples):.2f}%\n")
    print("\nP(Good) > P(Bad)\n")
    for x1, x2 in good_examples[:10]:
        print(f"p({x1}) > p({x2})")
    print("\nP(Good) == P(Bad)\n")
    for x1, x2 in error_examples[:10]:
        print(f"p({x1})==p({x2})")
