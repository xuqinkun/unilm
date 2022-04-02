# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import re
from layoutlmft.data.utils import normalize_bbox, merge_bbox, simplify_bbox

pattern_rar = "^(\s|\S)*\.((rar|zip|gz)(\.lock)?)$"
pattern_img = "^((\s|\S)*\.)?(png|jpg|jpeg)$"


def get_file_index(path_or_paths):
    file_dict = {}
    if isinstance(path_or_paths, str):
        path_or_paths = [path_or_paths]
    for path in path_or_paths:
        if osp.isfile(path):
            # 文件直接解析
            file_type, key = _parse_file(path)
            if key not in file_dict.keys():
                file_dict[key] = {}
            file_dict[key][file_type] = path
        else:
            # 遍历目录
            for file in os.listdir(path):
                if file.startswith(".") or "." not in file:
                    continue
                file_type, key = _parse_file(file)
                if key not in file_dict.keys():
                    file_dict[key] = {}
                file_dict[key][file_type] = file
    return file_dict


def _parse_file(filename):
    # 过滤掉压缩文件
    if re.compile(pattern_rar).match(filename.lower()):
        return "rar", None
    if os.sep in filename:
        filename = filename.rsplit(os.sep, 1)[-1]
    name, suffix = filename.rsplit(".", 1)
    if suffix == "json":
        if "ocr" in name:
            key = name.rsplit("-", 1)[0]
            file_type = "ocr"
        else:
            file_type = "tag"
            key = name
    elif suffix == 'csv':
        file_type = 'template'
        key = name.rsplit("-", 1)[0]
    elif re.compile(pattern_img).match(filename.lower()):
        file_type = "img"
        key = name
    else:
        file_type = "other"
        key = name
    return file_type, key


def get_lines(ocr_data):
    lines = []
    for _page in ocr_data["pages"]:
        for _table in _page['table']:
            if len(_table["lines"]) != 0:
                lines += _table["lines"]
            else:
                for cell in _table["form_blocks"]:
                    lines += cell["lines"]
    return lines


def load_json(filepath):
    with open(filepath, 'r', encoding="utf-8") as f:
        ocr_data = json.load(f)
    if type(ocr_data) == str:
        ocr_data = json.loads(ocr_data)
    return ocr_data


def parse_labels(labels):
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


def get_doc_items(tokenizer, lines, labels, label_map, image_size):
    entities = []
    entity_id_to_index_map = {}
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    tag_line_ids, id2label = parse_labels(labels)

    for line_id, line in enumerate(lines):
        if len(line["text"].strip()) == 0:
            continue

        tokenized_inputs, tags, label_name, entity_span = parse_text(tokenizer, line.copy(), image_size, line_id,
                                                                     tag_line_ids, id2label, label_map)

        if tags[0] != "O":
            entity_id_to_index_map[line_id] = len(entities)
            entities.append(
                {
                    "start": len(tokenized_doc["input_ids"]) + entity_span[0],
                    "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]) + entity_span[1],
                    "label": label_name.upper(),
                }
            )
        for i in tokenized_doc:
            tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
    return tokenized_doc, entities


def parse_text(tokenizer, line, image_size, line_id, tag_line_ids, id2label, label_map):
    text_length = 0
    ocr_length = 0
    bbox = []
    last_box = None

    text = line["text"]
    tokenized_inputs = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    sentence_tokens = []
    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
        if token_id == 6:
            bbox.append(None)
            sentence_tokens.append(None)
            continue
        text_length += offset[1] - offset[0]
        tmp_box = []
        tmp_tokens = []
        while ocr_length < text_length:
            ocr_word = line["char_candidates"].pop(0)
            tmp_tokens.append(ocr_word[0])
            box = line['char_polygons'].pop(0)
            ocr_length += len(
                tokenizer._tokenizer.normalizer.normalize_str(ocr_word[0].strip())
            )
            tmp_box.append(simplify_bbox(box))
        if len(tmp_box) == 0:
            tmp_box = last_box
        if len(tmp_tokens) != 0:
            sentence_tokens.append("".join(tmp_tokens))
        else:
            sentence_tokens.append(None)
        bbox.append(normalize_bbox(merge_bbox(tmp_box), image_size))
        last_box = tmp_box
    bbox = [
        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
        for i, b in enumerate(bbox)
    ]
    if line_id not in tag_line_ids:
        label_name = "O"
        tags = [label_name] * len(bbox)
        entity_span = None
    else:
        label_name = label_map[id2label[line_id]]
        tags = ["O"] * len(bbox)
        # 包含冒号的文本需要特殊处理
        if ":" in text:
            token_start = 0
            id_start = 0
            offset = 0
            # 遍历到最后一个:才停止
            for i, token in enumerate(sentence_tokens):
                if token:
                    if ":" == token:
                        # token是冒号则表示该token为分界点
                        token_start = i
                        id_start = offset
                    elif ":" in token:
                        # token包含冒号则表示该token为实体起点
                        token_start = i - 1
                        id_start = offset
                    offset += len(token)
                else:
                    id_start += 1
            try:
                if token_start + 1 < len(tags):
                    tags[token_start + 1] = f"B-{label_name.upper()}"
                if token_start + 2 < len(tags):
                    tags[token_start + 2:] = [f"I-{label_name.upper()}"] * (len(bbox) - token_start - 2)
                # 如果冒号在文本末尾，那么表明当前行文本是key，而下一行文本才是对应的value
                if token_start == len(tags) - 1:
                    tag_line_ids.add(line_id + 1)
                    id2label[line_id + 1] = id2label[line_id]
            except IndexError as e:
                raise e
            entity_span = (id_start, len(text))
        else:
            # 不包含引号则整行文本视作一个实体
            tags = [f"I-{label_name.upper()}"] * len(bbox)
            tags[0] = f"B-{label_name.upper()}"
            entity_span = (0, len(text))
    assert len(tags) == len(tokenized_inputs["input_ids"]), "Not equal"

    with open("/tmp/data.txt", "a") as f:
        for token, label in zip(sentence_tokens, tags):
            f.write("%s\t%s\n" % (token, label))
        f.write("\n")
    tokenized_inputs.update({"bbox": bbox, "labels": tags})

    return tokenized_inputs, tags, label_name, entity_span


def walk_dir(root_dir):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_type, _ = _parse_file(file)
            if file_type != "rar":
                file_list.append(osp.join(root, file))
            else:
                print("Skip ", file)
    return file_list


def update_ocr_index(file_dict, ocr_path):
    for key in file_dict:
        ocr_file = osp.join(ocr_path, key + ".json")
        if osp.exists(ocr_file) and osp.isfile(ocr_file):
            file_dict[key]["ocr"] = ocr_file