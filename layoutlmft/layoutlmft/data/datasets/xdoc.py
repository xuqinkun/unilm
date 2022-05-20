# -*- coding: utf-8 -*-
import json
import logging
import os
import random
from pathlib import Path

import datasets
from layoutlmft.data.utils import load_image, read_ner_label
from transformers import AutoTokenizer

from .utils import get_file_index, get_doc_items, get_lines, load_json, walk_dir, update_ocr_index
from .utils import ocr

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

LABEL_MAP = {
    "invoice": {"凭证号": "ID", "发票类型": "TAX_TYPE", "日期": "DATE", "客户名称": "CUSTOMER", "供用商名称": "SUPPLIER",
                "数量": "NUM", "税率": "TAX_RATE", "金额": "AMOUNT", "备注": "REMARK", "总金额": "TOTAL_AMOUNT", },
    "contract_entire": {"合同编号": "CONTRACT_ID", "客户名称": "FIRST_PARTY", "签订主体": "SECOND_PARTY",
                        "合同金额": "AMOUNT", "签订日期": "SIGN_DATE", "交货日期": "DELIVER_DATE",
                        '运输方式': "TRANSPORTATION", "产品名称": "PRODUCT_NAME", '签订地点': "SIGN_PLACE",
                        "付款方式": "PAYMENT_METHOD"},
    "contract": {"合同号": "CONTRACT_ID", "甲方": "FIRST_PARTY", "乙方": "SECOND_PARTY", "总金额": "AMOUNT", "日期": "DATE"},
    "receipt": {"金额": "AMOUNT", "日期": "DATE"},
    "voucher": {'编号': "ID", '科目': 'SUBJECT', '日期': "DATE", '金额': "AMOUNT", '摘要': "ABSTRACT"},
    "other": {'编号': "ID", '日期': "DATE", '金额': "AMOUNT", },
}

if "DEBUG" in os.environ:
    print(__file__.rsplit("/", 1)[0])


class XDocConfig(datasets.BuilderConfig):

    def __init__(self, lang, addtional_langs=None, **kwargs):
        super(XDocConfig, self).__init__(**kwargs)
        self.lang = lang
        self.addtional_langs = addtional_langs


class XDoc(datasets.GeneratorBasedBuilder):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __init__(self, **kwargs):
        self.data_dir = Path(kwargs['data_dir'])
        self.pred_only = kwargs['pred_only']
        self.is_tar_file = kwargs['is_tar_file']
        self.output_dir = Path(kwargs['output_dir'])
        self.label_map = LABEL_MAP[kwargs['doc_type']]
        self.label_names = list(self.label_map.values())
        self.ocr_path = None
        self.force_ocr = None
        if "ocr_path" in kwargs and kwargs['ocr_path']:
            self.ocr_path = Path(kwargs['ocr_path'])
            self.ocr_path.mkdir(exist_ok=True)
        if 'force_ocr' in kwargs and kwargs['force_ocr']:
            self.force_ocr = kwargs['force_ocr']
        self.labels = ["O"]
        for label in self.label_names:
            self.labels.append(f"B-{label}")
            self.labels.append(f"I-{label}")
        version = kwargs['version']
        if version:
            super(XDoc, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'], version=version)
        else:
            super(XDoc, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'])
        self.BUILDER_CONFIGS = [XDocConfig(name=f"x{kwargs['doc_type']}.{lang}", lang=lang) for lang in _LANG]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=self.labels
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=self.label_names),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.pred_only:
            if self.is_tar_file:
                eval_file = self.data_dir / "eval" / "eval.tar.gz"
                train_file = self.data_dir / "train" / "train.tar.gz"
            else:
                eval_file = walk_dir(self.data_dir.as_posix())
                train_file = []
        else:
            if self.is_tar_file:
                eval_file = self.data_dir / "eval" / "eval.tar.gz"
                train_file = self.data_dir / "train" / "train.tar.gz"
            else:
                train_file = walk_dir(self.data_dir / "train")
                eval_file = walk_dir(self.data_dir / "eval")

        data_dir = dl_manager.download_and_extract({
            "train": train_file.as_posix(), "eval": eval_file.as_posix()}
        )
        if self.output_dir:
            train_sample = self.output_dir / "train.csv"
            eval_sample = self.output_dir / "eval.csv"
            if train_sample.exists():
                os.remove(train_sample)
            if eval_sample.exists():
                os.remove(eval_sample)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_or_paths": data_dir["train"],
                    "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_or_paths": data_dir["eval"],
                    "split": "eval"}
            ),
        ]

    def _generate_examples(self, path_or_paths, split):
        """ Yields examples.
        这个方法将接收在前面的' _split_generators '方法中定义的' gen_kwargs '作为参数。
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)
        它负责打开给定的文件并从数据集生成元组(键，示例)
        key是不重要的，更多的是为了传承

        这里就是根据自己的数据集来整理
        """
        file_dict = get_file_index(path_or_paths)
        if self.ocr_path and not self.is_tar_file:
            update_ocr_index(file_dict, self.ocr_path)
        for key, file_group in file_dict.items():
            if 'img' not in file_group:
                print(f"Can't find img file of {key}")
                continue
            img_path = file_group['img']
            if 'ocr' not in file_group:
                if self.force_ocr:
                    ocr_data = ocr(img_path)
                    ocr_file = self.ocr_path / (key + ".json")
                    with ocr_file.open("w") as f_ocr:
                        json.dump(ocr_data, f_ocr)
                else:
                    print(f"Can't find ocr file of {key}")
                    continue
            else:
                ocr_data = load_json(file_group['ocr'])
            if ocr_data is None:
                print("Skip ", key, ",", file_group)
                continue
            image, image_shape = load_image(img_path)

            if 'tag' in file_group.keys():
                label_data = load_json(file_group['tag'])
                labels = read_ner_label(ocr_data, label_data)
            else:
                labels = None
            lines = get_lines(ocr_data)

            tokenized_doc, entities = get_doc_items(tokenizer=self.tokenizer, lines=lines, labels=labels,
                                                    label_map=self.label_map,
                                                    image_shape=image_shape, output_dir=self.output_dir, split=split)
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
                guid = img_path
                item.update(
                    {
                        "id": f"{guid}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                    }
                )
                yield f"{guid}_{chunk_id}", item
