# -*- coding: utf-8 -*-
import json
import logging
import os
import os.path as osp
import datasets
from layoutlmft.data.utils import load_image, read_ner_label, normalize_bbox, merge_bbox, simplify_bbox
from transformers import AutoTokenizer

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

LABEL_MAP = {"金额": "AMOUNT", "日期": "DATE"}


class XReceiptConfig(datasets.BuilderConfig):

    def __init__(self, lang, addtional_langs=None, **kwargs):
        super(XReceiptConfig, self).__init__(**kwargs)
        self.lang = lang
        self.addtional_langs = addtional_langs


class XReceipt(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [XReceiptConfig(name=f"xreceipt.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def __init__(self, **kwargs):
        super(XReceipt, self).__init__(**kwargs)
        self.data_dir = kwargs['data_dir']

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-AMOUNT", "B-DATE", "I-AMOUNT", "I-DATE"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["AMOUNT", "DATE"]),
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
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath": data_dir["test"],
            #         "split": "test"}
            # ),
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
            if file.startswith(".") or "." not in file:
                continue
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
            file_group = file_dict[key]
            if 'ocr' not in file_group.keys():
                print(key, file_group)
                exit(1)
            img_file = os.path.join(filepath, file_group['img'])
            ocr_filepath = os.path.join(filepath, file_group['ocr'])
            with open(ocr_filepath, 'r', encoding="utf-8") as f:
                ocr_data = json.load(f)
            if type(ocr_data) == str:
                ocr_data = json.loads(ocr_data)
            image, size = load_image(img_file)

            if 'tag' in file_group.keys():
                labels = read_ner_label(ocr_filepath, os.path.join(filepath, file_group['tag']))
            else:
                labels = None
            lines = []
            entity_id_to_index_map = {}
            for _page in ocr_data["pages"]:
                for _table in _page['table']:
                    if len(_table["lines"]) != 0:
                        lines += _table["lines"]
                    else:
                        for cell in _table["form_blocks"]:
                            lines += cell["lines"]
            entities = []
            tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
            tag_line_ids, id2label = _parse_labels(labels)
            guid = img_file
            for line_id, line in enumerate(lines):
                if len(line["text"].strip()) == 0:
                    continue
                tokenized_inputs = self.tokenizer(
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
                        ocr_word = line["char_candidates"].pop(0)
                        box = line['char_polygons'].pop(0)
                        ocr_length += len(
                            self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word[0].strip())
                        )
                        tmp_box.append(simplify_bbox(box))
                    if len(tmp_box) == 0:
                        tmp_box = last_box
                    bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                    last_box = tmp_box
                bbox = [
                    [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                    for i, b in enumerate(bbox)
                ]
                if line_id not in tag_line_ids:
                    label_name = "O"
                    tags = [label_name] * len(bbox)
                else:
                    label_name = LABEL_MAP[id2label[line_id]]
                    tags = [f"I-{label_name.upper()}"] * len(bbox)
                    tags[0] = f"B-{label_name.upper()}"
                tokenized_inputs.update({"bbox": bbox, "labels": tags})

                if tags[0] != "O":
                    entity_id_to_index_map[line_id] = len(entities)
                    entities.append(
                        {
                            "start": len(tokenized_doc["input_ids"]),
                            "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                            "label": label_name.upper(),
                        }
                    )
                for i in tokenized_doc:
                    tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
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
                        "id": f"{guid}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                    }
                )
                key = f"{guid}_{chunk_id}"
                # if len(tag_line_ids) == 0:
                #     item["labels"] = []
                yield key, item


def _parse_labels(labels):
    tag_line_ids = set()
    id2label = {}
    if labels:
        for label_name, words in labels:
            if len(words) == 0:
                continue
            line_id = words[0].line_id
            tag_line_ids.add(line_id)
            id2label[line_id] = label_name
    return tag_line_ids, id2label
