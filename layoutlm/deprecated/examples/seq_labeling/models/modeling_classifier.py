# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.models.layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config
from transformers.trainer import Trainer
from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import ResnetForImageTextMatching
import layoutlm.deprecated.examples.seq_labeling.data.sroie as sroie
from layoutlm.deprecated.examples.seq_labeling.data.data_collator import DataCollatorForClassifier
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.models.model_args import ModelArguments

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
    labels = {0: "covered", 1: "uncovered"}
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    batch_size = 8
    label_to_id = {v: k for k, v in labels.items()}

    data_collator = DataCollatorForClassifier(
        tokenizer,
        # pad_to_multiple_of=batch_size,
        padding="max_length",
        max_length=20,
        label_to_id=label_to_id,
        # device=device,
    )
    sampler = SequentialSampler(train_dataset)
    features = train_dataset.features
    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           sampler=sampler, collate_fn=data_collator)
    optimizer = Adam(model.parameters(), lr=0.00001)

    model.cuda()
    trainer = Trainer(model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=dataset['test'],
                      tokenizer=tokenizer,
                      )
    trainer.train()
