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
    max_seq_length: Optional[int] = None
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

        input_ids = [feature['input_ids'][:self.max_seq_length] for feature in features]
        bbox = [feature['bbox'][:self.max_seq_length] for feature in features]
        label = [feature['score'] for feature in features]
        batch = {
            "input_ids": input_ids,
            "bbox": bbox,
        }
        batch = self.tokenizer.pad(batch,
                                   max_length=self.max_seq_length,
                                   padding='max_length')

        batch['bbox'] = [bbox + [[0, 0, 0, 0]] * (self.max_seq_length - len(bbox)) for bbox in batch['bbox']]
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        batch['image'] = image
        batch['scores'] = torch.tensor(label, dtype=torch.float32)
        return batch
