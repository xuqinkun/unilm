# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint
from tqdm import tqdm
import layoutlm.deprecated.examples.seq_labeling.data.xdoc_perturbation_score as xdoc_perturbation
from layoutlm.deprecated.examples.seq_labeling.data.data_collator_score import DataCollatorForScore
from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import LayoutlmForImageTextMatching

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


def pad(input_ids, bbox, image, max_seq_len, device):
    # attention_mask = [1] * len(input_ids)
    # if len(input_ids) < max_seq_len:
    #     input_ids = input_ids + [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
    # if len(input_ids) > max_seq_len:
    #     input_ids = input_ids[:max_seq_len]
    # if len(bbox) < max_seq_len:
    #     bbox = bbox + [[0, 0, 0, 0]] * (max_seq_len - len(bbox))
    # if len(bbox) > max_seq_len:
    #     bbox = bbox[:max_seq_len]
    # if len(attention_mask) < max_seq_len:
    #     attention_mask = attention_mask + [0] * (max_seq_len - len(attention_mask))
    # if len(attention_mask) > max_seq_len:
    #     attention_mask = attention_mask[:max_seq_len]
    return {
        "input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
        "bbox": torch.tensor(bbox).unsqueeze(0).to(device),
        "image": torch.tensor(image).unsqueeze(0).to(device),
        # "attention_mask": torch.tensor(attention_mask).unsqueeze(0).to(device),
    }


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = LayoutLMv2Config.from_pretrained(model_args.model_name_or_path,
                                              name_or_path=model_args.model_name_or_path,
                                              num_labels=1)
    model = LayoutlmForImageTextMatching(config=config, max_seq_length=data_args.max_seq_length)
    version = None
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
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
        version='0.0.1',
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
    state_dict = torch.load(Path(last_checkpoint) / "pytorch_model.bin")
    state_dict.pop("proj.weight")
    state_dict.pop("proj.bias")
    state_dict.pop("classifier.weight")
    state_dict.pop("classifier.bias")
    num_good_lg_than_bad = 0
    num_eq = 0
    collator_fn = DataCollatorForScore(
        tokenizer=tokenizer,

    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

