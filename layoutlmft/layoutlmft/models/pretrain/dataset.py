# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class V2Dataset(Dataset):
    def __init__(self, encoded_inputs, match_images_label, device):
        self.all_images = encoded_inputs['image']
        self.all_input_ids = encoded_inputs['input_ids']
        self.all_attention_masks = encoded_inputs['attention_mask']
        self.all_bboxes = encoded_inputs['bbox']
        self.all_tokens = encoded_inputs['tokens']
        self.all_mvlm_mask = encoded_inputs['mvlm_mask']
        self.all_tia_mask = encoded_inputs['tia_mask']
        self.all_labels = encoded_inputs['labels']
        self.all_match_labels = torch.tensor(match_images_label, device=device)

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):
        return (
            self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_tokens[index],
            self.all_tia_mask[index],
            self.all_labels[index],
            self.all_match_labels[index],
            self.all_mvlm_mask[index]
        )

