# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint

import layoutlm.deprecated.examples.seq_labeling.data.xdoc_perturbation_score as xdoc_perturbation
from layoutlm.deprecated.examples.seq_labeling.data.data_collator_score import DataCollatorForScore
from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import LayoutlmForImageTextMatching

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


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
        x_good = {
            "text": good_example,
            "input_ids": good_inputs,
            "bbox": feature['good_bbox'],
            "image": feature['image'],
        }
        x_bad = {
            "text": bad_example,
            "input_ids": bad_inputs,
            "bbox": feature['bad_bbox'],
            "image": feature['image'],
        }
        eval_samples.append((x_good, x_bad))
    return eval_samples


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = LayoutLMv2Config.from_pretrained(model_args.model_name_or_path,
                                              name_or_path=model_args.model_name_or_path,
                                              num_labels=1)
    model = LayoutlmForImageTextMatching(config=config, max_seq_length=data_args.max_seq_length)
    version = None
    tokenizer = AutoTokenizer.from_pretrained(data_args.encoder_tokenizer, use_fast=True)

    dataset = load_dataset(
        path=Path(xdoc_perturbation.__file__).as_posix(),
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        name=f"x{data_args.doc_type}.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,
        doc_type=data_args.doc_type,
        cache_dir=data_args.data_cache_dir,
        pred_only=data_args.pred_only,
        is_tar_file=data_args.is_tar_file,
        ocr_path=data_args.ocr_path,
        force_ocr=data_args.force_ocr,
        version='0.0.4',
        output_dir=training_args.output_dir,
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

    max_eval_examples = 500
    eval_samples = convert_dataset_to_sents(eval_dataset)
    eval_indices = np.arange(len(eval_samples))
    np.random.shuffle(eval_indices)
    eval_samples = [eval_samples[i] for i in eval_indices[:max_eval_examples]]
    trainer.train(resume_from_checkpoint=last_checkpoint)
    error_examples = []
    good_examples = []
    eq_samples = []
    if training_args.local_rank in [0, -1] and training_args.do_eval:
        device = 'cuda:0'
        model.to(device)
        model.eval()
        for x_good, x_bad in tqdm(eval_samples):
            good_example = x_good.pop("text")
            bad_example = x_bad.pop("text")
            x_good = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in x_good.items()}
            x_bad = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in x_bad.items()}
            good_score = model(**x_good).item()
            bad_score = model(**x_bad).item()
            if good_score > bad_score:
                good_examples.append((good_example, bad_example))
            elif good_score < bad_score:
                error_examples.append((good_example, bad_example))
            else:
                eq_samples.append((good_example, bad_example))

        print(f"p(x_good)>p(x_bad): {100 * len(good_examples) / len(eval_samples):.2f}%")
        print("\nGood\tBad\n")
        for x1, x2 in good_examples[:10]:
            print(f"p({x1}) > p({x2})")
        print("\nGood\tBad\n")
        for x1, x2 in error_examples[:10]:
            print(f"p({x1})==p({x2})")
        print(f"{100 * len(eq_samples) / len(eval_samples):.2f}%")
