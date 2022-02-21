# -*- coding: utf-8 -*-
import json
import logging
import os

import datasets
from layoutlmft.data.utils import load_image, read_ner_label, normalize_bbox
from transformers import AutoTokenizer

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

_URLs = {  # 本地文件的路径
    'train': "/home/std2020/xqk/data/receipt/train/train.tar.gz",
    'validation': "/home/std2020/xqk/data/receipt/val/val.tar.gz"
}

LABEL_MAP = {"凭证号": "ID", "发票类型": "TAX_TYPE", "日期": "DATE", "客户名称": "CUSTOMER", "供用商名称": "SUPPLIER",
             "数量": "NUM", "税率": "TAX_RATE", "金额": "AMOUNT", "备注": "REMARK"}


# class XReceiptConfig(datasets.BuilderConfig):
#
#     def __init__(self, lang, addtional_langs=None, **kwargs):
#         super(XReceiptConfig, self).__init__(**kwargs)
#         self.lang = lang
#         self.addtional_langs = addtional_langs

class ReceiptConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ReceiptConfig, self).__init__(**kwargs)


class XReceipt(datasets.GeneratorBasedBuilder):
    # BUILDER_CONFIGS = [XReceiptConfig(name=f"xreceipt.{lang}", lang=lang) for lang in _LANG]
    BUILDER_CONFIGS = [
        ReceiptConfig(name="xreceipt", version=datasets.Version("1.0.0"), description="xreceipt dataset"),
    ]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-ID", "B-SUPPLIER", "B-CUSTOMER", "B-AMOUNT", "B-NUM", "B-TAX_TYPE",
                                   "B-DATE", "B-TAX_RATE", "B-REMARK", "I-ID", "I-SUPPLIER", "I-CUSTOMER", "I-AMOUNT",
                                   "I-NUM", "I-TAX_TYPE", "I-DATE", "I-TAX_RATE", "I-REMARK"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URLs)
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
                    "filepath": data_dir["validation"],
                    "split": "validation"}
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
        file_dict = {}
        for file in os.listdir(filepath):
            name, suffix = file.rsplit(".", 1)
            key = name
            if suffix == "json":
                if "ocr" in name:
                    key = name.rsplit("-", 1)[0]
                    file_type = "ocr"
                else:
                    file_type = "tag"
            else:
                file_type = "img"
            if key not in file_dict.keys():
                file_dict[key] = {}
            file_dict[key][file_type] = file

        for key in file_dict.keys():
            tag_filepath = os.path.join(filepath, file_dict[key]['tag'])
            img_file = os.path.join(filepath, file_dict[key]['img'])
            ocr_filepath = os.path.join(filepath, file_dict[key]['ocr'])
            with open(tag_filepath, 'r', encoding="utf-8") as f:
                tag_data = json.load(f)
            with open(ocr_filepath, 'r', encoding="utf-8") as f:
                ocr_data = json.load(f)
            if type(ocr_data) == str:
                ocr_data = json.loads(ocr_data)
            img_path = tag_data['imagePath']
            image, size = load_image(img_file)
            labels = read_ner_label(ocr_filepath, tag_filepath)
            ner_tags = []
            ocr_tokens = {}
            tag_line_ids = set()
            lines = []
            for _page in ocr_data["pages"]:
                for _table in _page['table']:
                    if len(_table["lines"]) != 0:
                        lines += _table["lines"]
                    else:
                        for cell in _table["form_blocks"]:
                            lines += cell["lines"]
            tokens = []
            bboxes = []
            id2label = {}
            for label_name, words in labels:
                if len(words) == 0:
                    continue
                line_id = words[0].line_id
                tag_line_ids.add(line_id)
                id2label[line_id] = label_name
            guid = img_file
            for line_id, line in enumerate(lines):
                word_idx = 0
                tokenized_inputs = self.tokenizer(
                    line["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )
                for token, box in zip(line["text"], line["char_polygons"]):
                    tokens.append(token)
                    bboxes.append(normalize_bbox(box, size))
                    if line_id in tag_line_ids:
                        if word_idx == 0:
                            ner_tags.append("B-" + LABEL_MAP[id2label[line_id]])
                        else:
                            ner_tags.append("I-" + LABEL_MAP[id2label[line_id]])
                    else:
                        ner_tags.append("O")
                    word_idx += 1
            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}
