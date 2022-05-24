# -*- coding: utf-8 -*-
import random
from dataclasses import dataclass
from typing import Optional, Union, Mapping

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForScore:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    label_to_id: Mapping = None

    device: str = None

    def __call__(self, features):
        image = None
        has_image_input = "image" in features[0]
        if has_image_input:
            image = torch.tensor([feature["image"] for feature in features], dtype=torch.float)
            for feature in features:
                del feature["image"]

        good_inputs = [feature['good_inputs'] for feature in features]
        good_bbox = [feature['good_bbox'] for feature in features]
        bad_inputs = [feature['bad_inputs'] for feature in features]
        bad_bbox = [feature['bad_bbox'] for feature in features]
        good_scores = [feature['good_label'] for feature in features]
        bad_scores = [feature['bad_label'] for feature in features]
        batch = {
            "input_ids": good_inputs + bad_inputs,
            "bbox": good_bbox + bad_bbox,
        }
        batch = self.tokenizer.pad(batch, max_length=self.max_length)
        max_seq_length = len(batch["input_ids"][0])
        batch['bbox'] = [bbox + [[0, 0, 0, 0]] * (max_seq_length - len(bbox)) for bbox in batch['bbox']]
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        batch['image'] = torch.cat((image, image), dim=0)
        batch['scores'] = torch.tensor(good_scores + bad_scores, dtype=torch.float32)
        return batch
