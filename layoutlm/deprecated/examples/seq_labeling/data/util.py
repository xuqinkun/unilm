# -*- coding: utf-8 -*-
import os
import json
import shutil
import random
import logging
import sys
sys.path.append(os.path.dirname(__file__))

from pathlib import Path
from layoutlmft.data.utils import normalize_bbox, merge_bbox, simplify_bbox
from txt.similar_words import word_dict, words

COVERED = 1
UNCOVERED = 0

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


def split_files(file_index):
    train_files = []
    eval_files = []
    test_files = []
    for k, item in file_index.items():
        if 'img' in item and 'ocr' in item and 'tag' in item:
            if random.random() < 0.9:
                train_files.append(item)
            else:
                eval_files.append(item)
        elif 'img' in item and 'ocr' in item:
            test_files.append(item)
        else:
            # Skip single file which is redundant
            if len(item) == 2:
                logger.info(f"Skip item {k}: {item}")
            pass
    return eval_files, test_files, train_files


def copy_files(files, dest_dir: Path):
    ocr_dir = dest_dir / "ocr"
    img_dir = dest_dir / "img"
    tag_dir = dest_dir / "tag"
    if ocr_dir.exists():
        shutil.rmtree(ocr_dir)
    if img_dir.exists():
        shutil.rmtree(img_dir)
    if tag_dir.exists():
        shutil.rmtree(tag_dir)
    ocr_dir.mkdir(exist_ok=True, parents=True)
    img_dir.mkdir(exist_ok=True, parents=True)
    tag_dir.mkdir(exist_ok=True, parents=True)

    for item in files:
        ocr_name = Path(item['ocr']).name
        img_name = Path(item['img']).name
        if 'img' not in item:
            continue
        ocr_dest_file = ocr_dir / ocr_name
        img_dest_file = img_dir / img_name
        ocr_dest_file.write_bytes(Path(item['ocr']).read_bytes())
        try:
            img_dest_file.write_bytes(Path(item['img']).read_bytes())
        except Exception as e:
            raise e
        if 'tag' in item:
            # Test dataset does not contain tag file
            tag_dest_file = tag_dir / ocr_name
            tag_dest_file.write_bytes(Path(item['tag']).read_bytes())


def get_file_index(src_dir, skip_dir):
    file_index = {}
    for root, dirs, files in os.walk(src_dir):
        if root.endswith(skip_dir):
            logger.info(f"Skip dir {root}")
            continue
        for file in files:
            full_path = Path(root) / file
            key = file.split(".")[0]
            if file.endswith("jpg"):
                type = 'img'
            else:
                with full_path.open("r") as f:
                    try:
                        json.load(f)
                        type = 'tag'
                    except:
                        type = 'ocr'
            if key not in file_index:
                file_index[key] = {}
            file_index[key][type] = full_path.as_posix()
    return file_index


def load_cache(_dir: Path):
    logger.info("Load cache from %s", _dir)
    img_files = sorted((_dir / 'img').glob("*.jpg"))
    tag_files = sorted((_dir / 'tag').glob("*.txt"))
    ocr_files = sorted((_dir / 'ocr').glob("*.txt"))
    file_index = []
    for img, tag, ocr in zip(img_files, tag_files, ocr_files):
        assert img.stem == tag.stem == ocr.stem
        file_index.append({
            "img": img.as_posix(),
            "tag": tag.as_posix(),
            "ocr": ocr.as_posix(),
        })
    return file_index


def read_json_file(path):
    src = Path(path)
    if not src.exists():
        return None
    with src.open("r", encoding='utf-8') as f:
        return json.load(f)


replace_token_prob = 0.1
delete_token_prob = 0.1
insert_token_prob = 0.1
print(f"replace_token_prob:{replace_token_prob}")
print(f"delete_token_prob:{delete_token_prob}")
print(f"insert_token_prob:{insert_token_prob}")


def get_sent_perturbation_word_level(tokenizer, line, n_samples, img_size):
    tokenized_inputs = tokenizer(
        line["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    input_ids = tokenized_inputs["input_ids"]
    offset_mapping = tokenized_inputs["offset_mapping"]
    if 'char_polygons' not in line:
        return None
    char_polygons = line['char_polygons']
    bbox = []
    for i, (start, end) in enumerate(offset_mapping):
        box = merge_bbox([simplify_bbox(b) for b in char_polygons[start: end]])
        bbox.append(normalize_bbox(box, size=img_size))
    dummy_inputs = [input_ids]
    dummy_bbox = [bbox]
    dummy_labels = [COVERED]

    all_special_ids = tokenizer.all_special_ids
    vocab = tokenizer.vocab
    id2word = {v: k for k, v in vocab.items()}
    vocab_size = len(words) - 1
    while len(dummy_inputs) < n_samples + 1 and len(input_ids) > 3:
        tmp_tokens = []
        tmp_box = []
        label = COVERED
        for token, box in zip(input_ids, bbox):
            tmp_box.append(box)
            if token in all_special_ids or token == 6:
                # Skip special ids
                tmp_tokens.append(token)
                continue
            prob = random.random()
            if prob < replace_token_prob:
                # Replace current token by a random token in vocab
                curr_word = id2word[token]
                if curr_word in word_dict:
                    similar_word_set = list(word_dict[curr_word])
                    rand_index = random.randint(0, len(similar_word_set) - 1)
                    while similar_word_set[rand_index] not in vocab:
                        rand_index = random.randint(0, len(similar_word_set) - 1)
                    tmp_tokens.append(vocab[similar_word_set[rand_index]])
                else:
                    tmp_tokens.append(token - 1)
                label = UNCOVERED
            elif prob < replace_token_prob + delete_token_prob:
                # Drop token
                tmp_box.pop(-1)
                label = UNCOVERED
            elif prob < replace_token_prob + delete_token_prob + insert_token_prob:
                tmp_tokens.append(token)
                # Insert a token
                rand_index = random.randint(0, vocab_size)
                rand_word = None
                while rand_index == token or rand_index in all_special_ids or rand_word not in vocab:
                    rand_index = random.randint(0, vocab_size)
                    rand_word = words[rand_index]
                tmp_tokens.append(vocab[rand_word])
                tmp_box.append(box)
                label = UNCOVERED
            else:
                tmp_tokens.append(token)
        if tmp_tokens not in dummy_inputs and len(tmp_tokens) > 0:
            dummy_labels.append(label)
            dummy_bbox.append(tmp_box)
            dummy_inputs.append(tmp_tokens)

    return dummy_inputs, dummy_bbox, dummy_labels
