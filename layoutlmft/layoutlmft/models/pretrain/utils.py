# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from pathlib import Path


def Process(processor, device, images, words, boxes, tokens, covers):
    encoded_inputs = {}
    encoded_input = processor(text=words, word_labels=covers, boxes=boxes, images=images,
                              padding="max_length", truncation=True, max_length=512)
    encoded_input2 = processor(text=tokens, word_labels=covers, boxes=boxes, images=images,
                               padding="max_length", truncation=True, max_length=512)

    encoded_inputs['image'] = torch.tensor(encoded_input['image'], device=device)

    #################################包含mask
    # 250001
    # 16（情况1）
    # 250001  250001
    # 6     16      6   3024（请况2）
    # 从头往后两个一起遍历，列表1和列表2，列表1为mask后的分词输出(指针a)，列表2为原始分词输出(指针b)
    # a和b同时往后走，
    #################################
    mask_tokens = encoded_input['input_ids']  # 用token的指针

    tokens = encoded_input2['input_ids']
    final_mask_tokens = np.copy(encoded_input2['input_ids'])  # 深拷贝

    mvlm_mask = np.zeros_like(tokens)
    tia_mask = np.ones_like(tokens)  # cover
    ##################################################
    for batch in range(len(tokens)):
        mask_token = mask_tokens[batch]
        token = tokens[batch]
        i = 0  # mask token
        j = 0  # token
        while i < len(mask_token) and j < len(token):
            if mask_token[i] == 250001 and token[j] != 6:
                final_mask_tokens[batch][j] = 250001
                mvlm_mask[batch][j] = 1
                tia_mask[batch][j] = 0
            elif mask_token[i] == 250001 and token[j] == 6:
                tia_mask[batch][j] = 0
                mvlm_mask[batch][j] = 1
                j += 1
                final_mask_tokens[batch][j] = 250001
                tia_mask[batch][j] = 0
                mvlm_mask[batch][j] = 1
            i += 1
            j += 1

    #################################
    encoded_inputs['input_ids'] = torch.tensor(final_mask_tokens, device=device)
    encoded_inputs['attention_mask'] = torch.tensor(encoded_input2['attention_mask'], device=device)
    encoded_inputs['bbox'] = torch.tensor(encoded_input2['bbox'], device=device)
    encoded_inputs['tokens'] = torch.tensor(tokens, device=device)
    encoded_inputs['labels'] = torch.tensor(encoded_input2['labels'], device=device)
    encoded_inputs['mvlm_mask'] = torch.tensor(mvlm_mask, device=device)
    encoded_inputs['tia_mask'] = torch.tensor(tia_mask, device=device)

    return encoded_inputs


def convert_examples_to_features(
        tokens, words, bboxes, covers, max_seq_len=512,
):
    final_words = []
    final_boxes = []
    final_tokens = []
    final_covers = []

    for token, word, bbox, cover in zip(tokens, words, bboxes, covers):
        padding_length = (max_seq_len - 2) - len(word)  # padding 到 510

        word += ["<pad>"] * padding_length
        token += ["<pad>"] * padding_length

        cover += [0] * padding_length

        bbox += [[0, 0, 0, 0]] * padding_length

        final_tokens.append(token)
        final_words.append(word)
        final_boxes.append(bbox)
        final_covers.append(cover)

    return final_tokens, final_words, final_boxes, final_covers


def read_lines(data_dir, data_root_dir:Path):
    data_dict = {}
    with open(data_dir, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                continue
            else:
                tokens = line[:-1].split("\t")
                img_file = tokens.pop(-1)
                file_splits = img_file.split(os.sep)
                last_dir, filename = file_splits[-2], file_splits[-1]
                if filename not in data_dict:
                    data_dict[filename] = {}
                    data_dict[filename]["tokens"] = []
                    data_dict[filename]["labels"] = []
                    data_dict[filename]["bbox"] = []
                    data_dict[filename]["actual_bbox"] = []
                data_dict[filename]["tokens"].append(tokens[0])
                data_dict[filename]["labels"].append(tokens[1])
                data_dict[filename]["bbox"].append(tokens[2])
                data_dict[filename]["actual_bbox"].append(tokens[3])
                new_img_file = data_root_dir / last_dir / filename
                data_dict[filename]['actual_path'] = new_img_file
    return data_dict