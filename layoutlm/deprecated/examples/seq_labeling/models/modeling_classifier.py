# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
from pathlib import Path

from datasets import load_dataset
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from transformers.trainer import Trainer
from transformers.trainer_utils import get_last_checkpoint

import layoutlm.deprecated.examples.seq_labeling.data.sroie as sroie
from layoutlm.deprecated.examples.seq_labeling.data.data_collator import DataCollatorForClassifier
from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import ResnetForImageTextMatching
from seqeval.metrics import f1_score, recall_score, precision_score

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
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = 'cpu'
    model = ResnetForImageTextMatching(config=config, device=device)
    # version='2.0.0'
    version = None
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    dataset = load_dataset(
        path=Path(sroie.__file__).as_posix(),
        data_dir='/home/std2020/xuqinkun/data/sroie',
        tokenizer=tokenizer,
        version=version,
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
    id2label = {0: "covered", 1: "uncovered"}
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    label_to_id = {v: k for k, v in id2label.items()}

    data_collator = DataCollatorForClassifier(
        tokenizer,
        # pad_to_multiple_of=batch_size,
        padding="max_length",
        max_length=data_args.max_seq_length,
        label_to_id=label_to_id,
        # device=device,
    )
    sampler = SequentialSampler(train_dataset)
    features = train_dataset.features

    def compute_metrics(p):
        logits, labels = p
        predictions = np.argmax(logits, axis=2).tolist()
        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        y_pred = [p for preds in true_predictions for p in preds]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        y_true = [l for labels in true_labels for l in labels]
        num_correct = sum([1 if p == l else 0 for p, l in zip(y_pred, y_true)])
        return {"accuracy": num_correct/len(y_true)}

    trainer = Trainer(model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=test_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      )
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset=test_dataset)
        for k, v in metrics.items():
            print(f"{k}:{v}")
    if training_args.do_predict:
        trainer.predict(test_dataset=test_dataset)
