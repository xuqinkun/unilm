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

import layoutlm.deprecated.examples.seq_labeling.data.xdoc_perturbation as xdoc_perturbation
from layoutlm.deprecated.examples.seq_labeling.data.data_collator import DataCollatorForClassifier
from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import LayoutlmForImageTextMatching

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = LayoutLMv2Config.from_pretrained(model_args.model_name_or_path,
                                              name_or_path=model_args.model_name_or_path,
                                              num_labels=2)
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
        version=data_args.version,
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

    max_train_samples = data_args.max_train_samples if data_args.max_train_samples else len(train_dataset)
    max_eval_samples = data_args.max_val_samples if data_args.max_val_samples else len(eval_dataset)
    train_indices = np.arange(len(train_dataset))
    eval_indices = np.arange(len(eval_dataset))
    np.random.shuffle(train_indices)
    np.random.shuffle(eval_indices)
    train_dataset = train_dataset.select(train_indices[:max_train_samples])
    eval_dataset = eval_dataset.select(eval_indices[:max_eval_samples])
    features = train_dataset.features['label'].names
    id2label = {i: l for i, l in enumerate(features)}
    label_to_id = {v: k for k, v in id2label.items()}
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # else:
    #     device = 'cpu'

    data_collator = DataCollatorForClassifier(
        tokenizer,
        # pad_to_multiple_of=batch_size,
        padding="max_length",
        max_length=data_args.max_seq_length,
        label_to_id=label_to_id,
        # device=device,
    )

    # model.to(device)


    def compute_metrics(p):
        logits, labels = p
        y_pred = np.argmax(logits, axis=-1).tolist()
        y_true = np.squeeze(labels, axis=-1)
        num_correct = sum([1 if p == l else 0 for p, l in zip(y_pred, y_true)])
        return {"accuracy": num_correct / len(y_true)}


    trainer = Trainer(model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"{k}:{v}")
    if training_args.do_predict:
        trainer.predict(test_dataset=eval_dataset)
