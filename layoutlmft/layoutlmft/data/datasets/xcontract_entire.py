# -*- coding: utf-8 -*-
import logging
import os
import os.path as osp

import datasets
from layoutlmft.data.utils import load_image, read_ner_label
from transformers import AutoTokenizer

from .utils import get_file_index, get_doc_items, get_lines, load_json

if "DEBUG" in os.environ:
    print(__file__.rsplit("/", 1)[0])

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

LABEL_MAP = {"合同编号": "CONTRACT_ID", "客户名称": "FIRST_PARTY", "签订主体": "SECOND_PARTY", "合同金额": "AMOUNT",
             "签订日期": "SIGN_DATE",
             "交货日期": "DELIVER_DATE",
             '运输方式': "TRANSPORTATION", "产品名称": "PRODUCT_NAME", '签订地点': "SIGN_PLACE", "付款方式": "PAYMENT_METHOD"}


class XContractEntireConfig(datasets.BuilderConfig):

    def __init__(self, lang, addtional_langs=None, **kwargs):
        super(XContractEntireConfig, self).__init__(**kwargs)
        self.lang = lang
        self.addtional_langs = addtional_langs


class XContractEntire(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [XContractEntireConfig(name=f"xcontract_entire.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __init__(self, **kwargs):
        super(XContractEntire, self).__init__(**kwargs)
        self.data_dir = kwargs['data_dir']

    def _info(self):
        label_names = list(LABEL_MAP.values())
        labels = ["O"]
        for label in label_names:
            labels.append(f"B-{label}")
            labels.append(f"I-{label}")
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=labels
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=label_names),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_file = osp.join(self.data_dir, "train", "train.tar.gz")
        eval_file = osp.join(self.data_dir, "eval", "eval.tar.gz")
        print(train_file, eval_file)
        data_dir = dl_manager.download_and_extract({
            "train": train_file, "eval": eval_file}
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["eval"],
                    "split": "eval"}
            ),
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples.
        这个方法将接收在前面的' _split_generators '方法中定义的' gen_kwargs '作为参数。
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)
        它负责打开给定的文件并从数据集生成元组(键，示例)
        key是不重要的，更多的是为了传承

        这里就是根据自己的数据集来整理
        """
        file_dict = get_file_index(filepath)

        for key, file_group in file_dict.items():
            if 'ocr' not in file_group.keys():
                exit(1)
            ocr_data = load_json(os.path.join(filepath, file_group['ocr']))
            image, image_size = load_image(os.path.join(filepath, file_group['img']))

            if 'tag' in file_group.keys():
                label_data = load_json(os.path.join(filepath, file_group['tag']))
                labels = read_ner_label(ocr_data, label_data)
            else:
                labels = None
            lines = get_lines(ocr_data)

            tokenized_doc, entities = get_doc_items(self.tokenizer, lines, labels, LABEL_MAP, image_size)

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

                item.update(
                    {
                        "id": f"{key}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                    }
                )
                yield f"{key}_{chunk_id}", item
