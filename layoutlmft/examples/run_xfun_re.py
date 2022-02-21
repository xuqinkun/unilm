#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys

from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
import torch
from torch.cuda.amp import autocast
from layoutlmft.models.layoutlmv2 import LayoutLMv2ForRelationExtraction
from datasets import ClassLabel, load_dataset

import layoutlmft.data.datasets.xfun
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft.data.data_args import XFUNDataTrainingArguments
from layoutlmft.data.data_collator import DataCollatorForKeyValueExtraction
from layoutlmft.evaluation import re_score
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import XfunReTrainer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, XFUNDataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    datasets = load_dataset(
        os.path.abspath(layoutlmft.data.datasets.xfun.__file__),
        f"xfun.{data_args.lang}",
        additional_langs=data_args.additional_langs,
        keep_in_memory=True,

    )
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "input_ids"
    label_column_name = "labels"

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForRelationExtraction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    def compute_metrics(p):
        pred_relations, gt_relations = p
        score = re_score(pred_relations, gt_relations, mode="boundaries")
        return score

    # Initialize our Trainer
    trainer = XfunReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        test_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=8,
            collate_fn=data_collator,
            pin_memory=True,
            drop_last=False,
            num_workers=0
        )
        do_predict(tokenizer.get_vocab(), test_dataloader, model)


import json
from layoutlmft.data.datasets.xfun import *

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]


def do_predict(vocab, dataloader, model):
    re_labels = None
    pred_relations = None
    entities = None
    chunk_ids = dataloader.dataset.data[2]
    id_to_word = {v: k for k, v in vocab.items()}
    text_map, index_to_entityid = _generate_examples([[
        "/home/std2020/.cache/huggingface/datasets/downloads/76a7c640a6347d1911a8fd117e3e18b7987eb1243c2e6e2b7cb39df9fa6e6cb2"
        ,
        "/home/std2020/.cache/huggingface/datasets/downloads/extracted/4e0768437b98d410c5c75e04de326344330227eae2bfac2d508362e17fe9247a"]])
    for step, inputs in enumerate(dataloader):
        inputs = _prepare_inputs(model.device, inputs)
        with torch.no_grad():
            with autocast():
                outputs = model(**inputs)
        labels = tuple(inputs.get(name) for name in ['labels', 'relations'])
        chunk_id = chunk_ids[step]
        with open(os.path.join("/tmp/outputs", str(chunk_id) + ".json")) as f:
            val_json = json.load(f)

        re_labels = labels[1] if re_labels is None else re_labels + labels[1]
        pred_relations = (
            outputs.pred_relations if pred_relations is None else pred_relations + outputs.pred_relations
        )
        entities = outputs.entities if entities is None else entities + outputs.entities
    for i, rel_list in enumerate(pred_relations):
        _map = text_map[i]
        index_to_id = index_to_entityid[i]
        entity = entities[i]
        for rel in rel_list:
            head_id = rel["head_id"]
            head_start = rel["head"][0]
            head_end = rel["head"][1]
            tail_id = rel["tail_id"]
            tail_start = rel["tail"][0]
            tail_end = rel["tail"][1]
            assert entity['start'][head_id] == head_start, \
                "head start error %d %d" % (entity['start'][head_id], head_start)
            assert entity['end'][head_id] == head_end, \
                "head end error %d %d" % (entity['end'][head_id], head_end)
            assert entity['start'][tail_id] == tail_start, \
                "tail start error %d %d" % (entity['start'][tail_id], tail_start)
            assert entity['end'][tail_id] == tail_end, \
                "tail end error %d %d" % (entity['end'][tail_id], tail_end)
            head_type = rel["head_type"]
            tail_type = rel["tail_type"]

            if head_id not in index_to_id.keys():
                print("head_id[%d] not found" % head_id)
                continue
            if tail_id not in index_to_id.keys():
                print("tail_id[%d] not found" % tail_id)
                continue

            head_index = index_to_id[head_id]
            tail_index = index_to_id[tail_id]

            if head_index not in _map.keys():
                print("head_index[%d] not found" % head_index)
                continue

            if tail_index not in _map.keys():
                print("tail_index[%d] not found" % tail_index)
                continue

            head = _map[head_index]
            tail = _map[tail_index]

            if head_type == 1 and tail_type == 2:
                print("%s->%s" % (head, tail))
            elif head_type == 2 and tail_type == 1:
                print("%s->%s" % (tail, head))



def _prepare_inputs(device, inputs):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if hasattr(v, "to") and hasattr(v, "device"):
            inputs[k] = v.to(device)
    return inputs


def _generate_examples(filepaths):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    text_map = {}
    index_to_entity_id = {}
    for filepath in filepaths:
        logger.info("Generating examples from = %s", filepath)
        with open(filepath[0], "r") as f:
            data = json.load(f)

        for doc in data["documents"]:
            doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
            image, size = load_image(doc["img"]["fpath"])
            document = doc["document"]
            id = int(doc["id"].split("_")[-1])
            tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
            entities = []
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()
            index_to_entity_id[id] = {}
            text_map[id] = {line['id']: line['text'] for line in document}
            for line in document:
                if len(line["text"]) == 0:
                    empty_entity.add(line["id"])
                    continue
                id2label[line["id"]] = line["label"]
                relations.extend([tuple(sorted(l)) for l in line["linking"]])
                tokenized_inputs = tokenizer(
                    line["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )
                text_length = 0
                ocr_length = 0
                bbox = []
                last_box = None
                for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                    if token_id == 6:
                        bbox.append(None)
                        continue
                    text_length += offset[1] - offset[0]
                    tmp_box = []
                    while ocr_length < text_length:
                        ocr_word = line["words"].pop(0)
                        ocr_length += len(
                            tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                        )
                        tmp_box.append(simplify_bbox(ocr_word["box"]))
                    if len(tmp_box) == 0:
                        tmp_box = last_box
                    bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                    last_box = tmp_box
                bbox = [
                    [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                    for i, b in enumerate(bbox)
                ]
                if line["label"] == "other":
                    label = ["O"] * len(bbox)
                else:
                    label = [f"I-{line['label'].upper()}"] * len(bbox)
                    label[0] = f"B-{line['label'].upper()}"
                tokenized_inputs.update({"bbox": bbox, "labels": label})
                if label[0] != "O":
                    entity_id_to_index_map[line["id"]] = len(entities)
                    index_to_entity_id[id][len(entities)] = line["id"]
                    entities.append(
                        {
                            "start": len(tokenized_doc["input_ids"]),
                            "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                            "label": line["label"].upper(),
                        }
                    )
                for i in tokenized_doc:
                    tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
            relations = list(set(relations))
            relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
            kvrelations = []

            for rel in relations:
                pair = [id2label[rel[0]], id2label[rel[1]]]
                if pair == ["question", "answer"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                    )
                elif pair == ["answer", "question"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                    )
                else:
                    continue

            def get_relation_span(rel):
                bound = []
                for entity_index in [rel["head"], rel["tail"]]:
                    bound.append(entities[entity_index]["start"])
                    bound.append(entities[entity_index]["end"])
                return min(bound), max(bound)

            relations = sorted(
                [
                    {
                        "head": rel["head"],
                        "tail": rel["tail"],
                        "start_index": get_relation_span(rel)[0],
                        "end_index": get_relation_span(rel)[1],
                    }
                    for rel in kvrelations
                ],
                key=lambda x: x["head"],
            ) # entity_id_to_index_map
            chunk_size = 512
            for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                item = {}
                for k in tokenized_doc:
                    item[k] = tokenized_doc[k][index: index + chunk_size]
                entities_in_this_span = []
                global_to_local_map = {}
                for entity_id, entity in enumerate(entities):
                    if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                    ):
                        entity["start"] = entity["start"] - index
                        entity["end"] = entity["end"] - index
                        global_to_local_map[entity_id] = len(entities_in_this_span)
                        entities_in_this_span.append(entity)
                relations_in_this_span = []
                for relation in relations:
                    if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                    ):
                        relations_in_this_span.append(
                            {
                                "head": global_to_local_map[relation["head"]],
                                "tail": global_to_local_map[relation["tail"]],
                                "start_index": relation["start_index"] - index,
                                "end_index": relation["end_index"] - index,
                            }
                        )

                item.update(
                    {
                        "id": f"{doc['id']}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                        "relations": relations_in_this_span,
                    }
                )
    return text_map, index_to_entity_id


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
