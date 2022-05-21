# -*- coding: utf-8 -*-
import random
from dataclasses import dataclass
from typing import Optional, Union, Mapping

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForClassifier:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    label_to_id: Mapping = None
    # device: str = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"

        # perturb(self.tokenizer.vocab, self.tokenizer.all_special_ids, features, self.label_to_id)
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        image = None
        box_name = "bbox"
        has_image_input = "image" in features[0]
        has_bbox_input = box_name in features[0]
        if has_image_input:
            image = torch.tensor([feature["image"] for feature in features], dtype=torch.float)
            for feature in features:
                del feature["image"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            batch[label_name] = [[label] + [self.label_pad_token_id] * (sequence_length - 1) for label in labels]
            if has_bbox_input:
                batch[box_name] = [[bbox] + [[0, 0, 0, 0]] * (sequence_length - 1) for bbox in batch[box_name]]
        else:
            batch[label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch[box_name] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch[box_name]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        if has_image_input:
            batch["image"] = image
        return batch


def perturb(vocab: dict, all_special_ids: list, features, label_to_id):
    # Perturbation for image text alignment and 15% uncovered,
    # among which 30% random line, 30% random image and 40% random bbox
    it_uncovered_prob = 0.15
    replace_text_prob = 0.8
    replace_token_prob = 0.7
    replace_line_prob = 0.9
    sample_size = len(features)
    for i, feature in enumerate(features):
        feature['ner_tags'] = [label_to_id['covered']] * len(feature['ner_tags'])
        if random.random() < it_uncovered_prob:
            continue
        # 85%概率为uncovered image text
        prob = random.random()
        if prob < replace_text_prob:
            # 80%概率进行文本替换
            if random.random() < replace_line_prob:
                # 90%替换单个token
                for j in range(len(feature['input_ids'])):
                    if feature['input_ids'][j] in all_special_ids or random.random() >= replace_token_prob:
                        continue
                    feature['ner_tags'][j] = label_to_id['uncovered']
                    # replace_token_prob概率用同字典中的其他token进行替换
                    new_token = random.randint(0, len(vocab) - 1)
                    while feature['input_ids'][j] == new_token or new_token in all_special_ids:
                        new_token = random.randint(0, len(vocab) - 1)
                    feature['input_ids'][j] = new_token
            else:
                # 10%概率用其他文档文本替换
                new_id = random.randint(0, sample_size - 1)
                while new_id == i:
                    new_id = random.randint(0, sample_size - 1)
                feature['input_ids'] = features[new_id]['input_ids'].copy()
                feature['ner_tags'] = [label_to_id['uncovered']] * len(feature['ner_tags'])
        else:
            # 剩下20%概率替换bbox
            p = random.random()
            bbox = feature['bboxes']
            for j in range(len(bbox)):
                if p < 0.5:
                    if len(bbox) == 1:
                        continue
                    feature['ner_tags'][j] = label_to_id['uncovered']
                    new_j = random.randint(0, len(bbox) - 1)
                    while j == new_j:
                        new_j = random.randint(0, len(bbox) - 1)
                    bbox[j] = bbox[new_j]
                elif p < 0.7:
                    if j > 0:
                        feature['ner_tags'][j] = label_to_id['uncovered']
                        bbox[j] = bbox[j - 1]
                elif p < 0.9:
                    if j < len(bbox) - 1:
                        feature['ner_tags'][j] = label_to_id['uncovered']
                        bbox[j] = bbox[j + 1]
                else:
                    feature['ner_tags'][j] = label_to_id['uncovered']
                    bbox[j] = [0, 0, 0, 0]
