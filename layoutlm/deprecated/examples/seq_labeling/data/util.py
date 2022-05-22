# -*- coding: utf-8 -*-
import os
import json
import shutil
import random
import logging
from pathlib import Path
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
