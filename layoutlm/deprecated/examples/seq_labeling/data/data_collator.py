# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Union

import torch

from detectron2.structures import ImageList
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

@dataclass
class DataCollatorForClassifier:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "ner_tags"

        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        lines = [feature["words"] for feature in features]
        image = None
        has_image_input = "image" in features[0]
        has_bbox_input = "bboxes" in features[0]
        if has_image_input:
            image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            for feature in features:
                del feature["image"]
                del feature['words']
                del feature['id']
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
            batch[label_name] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bboxes"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bboxes"]]
        else:
            batch[label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bboxes"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bboxes"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        if has_image_input:
            batch["image"] = image
        return batch
