# -*- coding: utf-8 -*-
# coding=utf-8
import json
import os
from pathlib import Path

import datasets
from layoutlmft.data.utils import load_image, simplify_bbox, normalize_bbox, merge_bbox

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{2019,
   title={ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
   url={http://dx.doi.org/10.1109/ICDAR.2019.00244},
   DOI={10.1109/icdar.2019.00244},
   journal={2019 International Conference on Document Analysis and Recognition (ICDAR)},
   publisher={IEEE},
   author={Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
   year={2019},
   month={Sep}
}
"""
_DESCRIPTION = """\
https://arxiv.org/abs/2103.10213
"""


class SroieConfig(datasets.BuilderConfig):
    """BuilderConfig for SROIE"""

    def __init__(self, **kwargs):
        """BuilderConfig for SROIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SroieConfig, self).__init__(**kwargs)


class Sroie(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SroieConfig(name="sroie", version=datasets.Version("1.0.0"), description="SROIE dataset"),
    ]

    def __init__(self, **kwargs):
        if 'version' in kwargs:
            super(Sroie, self).__init__(cache_dir=kwargs['cache_dir'],
                                        name=kwargs['name'],
                                        version=kwargs['version'])
        else:
            super(Sroie, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'])
        self.data_dir = Path(kwargs['data_dir']) / 'sroie.zip'
        self.tokenizer = kwargs['tokenizer']

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS",
                                   "B-TOTAL", "I-TOTAL"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://arxiv.org/abs/2103.10213",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        downloaded_file = dl_manager.download_and_extract(self.data_dir.as_posix())
        # move files from the second URL together with files from the first one.
        dest = Path(downloaded_file) / "sroie"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest / "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest / "test"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "tagged")
        img_dir = os.path.join(filepath, "images")
        for doc_id, fname in enumerate(sorted(os.listdir(img_dir))):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, fname)
            image, size = load_image(image_path)
            boxes = [normalize_bbox(simplify_bbox(box), size) for box in data["bbox"]]
            words, labels = data["words"], data["labels"]
            for i, (word, label, box) in enumerate(zip(words, labels, boxes)):
                ids = self.tokenizer(word, add_special_tokens=False, )['input_ids']
                boxes = [box] * len(ids)
                if label.startswith('B'):
                    _, label_name = label.split('-')
                    labels = [f'I-{label_name}'] * len(ids)
                    labels[0] = f'B-{label_name}'
                else:
                    labels = [label] * len(ids)
                guid = f'{doc_id}-{i}'
                yield guid, {"id": str(guid),
                             "words": data["words"],
                             "input_ids": ids,
                             "bboxes": boxes,
                             "ner_tags": labels,
                             "image": image,
                             }


from util import read_json_file, get_file_index
from nltk import word_tokenize, sent_tokenize
from layoutlmft.data.datasets.utils import ocr
from transformers import AutoTokenizer


def get_label_name(label):
    return label.split("-")[-1]


def generate_samples(split_dir):
    img_dir = split_dir / "images"
    tag_dir = split_dir / "tagged"
    img_files = sorted(img_dir.glob("*.jpg"))
    tag_files = sorted(tag_dir.glob("*.json"))
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    for i, (img_file, tag_file) in enumerate(zip(img_files, tag_files)):
        assert img_file.stem == tag_file.stem
        key = img_file.stem
        item = file_index[key]
        assert 'img' in item
        lines = []
        ocr_data = ocr(item['img'])
        for page in ocr_data['pages']:
            for table in page['table']:
                if table['type']:
                    # form blocks
                    for form in table['form_blocks']:
                        lines += form['lines']
                else:
                    lines += table['lines']
        word_spans = []
        word_tag = read_json_file(tag_file)
        words = word_tag['words']
        bbox = word_tag['bbox']
        word_labels = word_tag["labels"]
        start, end = 0, 0
        while start < len(word_labels):
            curr_label = get_label_name(word_labels[start])
            while end < len(word_labels) and curr_label == get_label_name(word_labels[end]):
                end += 1
            word_spans.append((start, end, curr_label))
            start = end
        sents = []
        sent_labels = []
        sent_bbox = []
        for span in word_spans:
            start, end, label = span
            sent = " ".join(words[start: end])
            sents.append(sent)
            sent_bbox.append(bbox[start: end])
            sent_labels.append([label] * (end - start))
        all_bbox = [b for line in lines for b in line['char_polygons']]
        all_text = [line["text"] for line in lines]
        doc = "".join(all_text)

        global_offset = 0
        image, img_size = load_image(img_file)
        for j, (sent, labels, bbox_list) in enumerate(zip(sents, sent_labels, sent_bbox)):
            guid = f'train-{i}-{j}'
            tokenized_input = tokenizer(sent,
                                        add_special_tokens=False,
                                        return_offsets_mapping=True,
                                        return_attention_mask=False, )
            input_ids = tokenized_input["input_ids"]
            offset_mapping = tokenized_input['offset_mapping']

            # global_x_left, global_x_right = bbox_list[0][0], bbox_list[-1][2]
            # y_up_avg = sum([b[1] for b in bbox_list])/len(bbox_list)
            # width_per_char = round((global_x_right - global_x_left)/len(sent))
            # avg_height = sum([b[3] - b[1] for b in bbox_list]) / len(bbox_list)
            bbox = []
            # for offset in offset_mapping:
            #     x_left = global_x_left + offset[0] * width_per_char
            #     x_right = x_left + width_per_char
            #     y_up = y_up_avg
            #     y_bottom = y_up + avg_height
            #     box = [x_left, y_up, x_right, y_bottom]
            #     bbox.append(box)
            for offset in offset_mapping:
                tmp_box = []
                start, end = offset
                for word_idx in range(start, end, 1):
                    global_start = global_offset + word_idx
                    head, tail = sent[start], sent[end - 1]
                    if doc[global_start] == sent[word_idx]:
                        tmp_box.append(simplify_bbox(all_bbox[global_start]))
                    else:
                        while doc[global_start] != sent[word_idx]:
                            global_offset += 1
                            global_start = global_offset + word_idx
                bbox.append(merge_bbox(tmp_box))
            global_offset += len(sent)


if __name__ == '__main__':
    root_dir = Path("/home/std2020/xuqinkun/data/sroie")
    sroie_words_dir = root_dir / 'sroie_words'
    sroie_raw_dir = root_dir / 'sroie_raw'
    file_index = get_file_index(sroie_raw_dir, 'processed')
    generate_samples(sroie_words_dir / "train")
