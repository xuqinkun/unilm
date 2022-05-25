# -*- coding: utf-8 -*-
import os
import json
import logging
import datasets
import numpy as np
from pathlib import Path
from layoutlmft.data.datasets.utils import (
    walk_dir,
    get_file_index,
    update_ocr_index,
    ocr,
    load_json,
    get_lines,
)
from layoutlmft.data.utils import load_image

from layoutlm.deprecated.examples.seq_labeling.data.util import (
    COVERED,
    UNCOVERED,
    get_sent_perturbation_word_level
)

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

if "DEBUG" in os.environ:
    print(__file__.rsplit("/", 1)[0])


class XConfig(datasets.BuilderConfig):

    def __init__(self, lang, addtional_langs=None, **kwargs):
        super(XConfig, self).__init__(**kwargs)
        self.lang = lang
        self.addtional_langs = addtional_langs


class XDocPerturbationScore(datasets.GeneratorBasedBuilder):

    def __init__(self, **kwargs):
        self.data_dir = Path(kwargs['data_dir'])
        self.pred_only = kwargs['pred_only']
        self.is_tar_file = kwargs['is_tar_file']
        self.output_dir = Path(kwargs['output_dir'])
        self.label_names = [COVERED, UNCOVERED]
        self.tokenizer = kwargs['tokenizer']
        if 'max_train_samples' in kwargs:
            self.max_train_samples = kwargs['max_train_samples']
        if 'max_eval_samples' in kwargs:
            self.max_eval_samples = kwargs['max_eval_samples']
        if 'multiple_of_neg_samples' in kwargs:
            self.multiple_of_neg_samples = kwargs['multiple_of_neg_samples']
        self.ocr_path = None
        self.force_ocr = None
        if "ocr_path" in kwargs and kwargs['ocr_path']:
            self.ocr_path = Path(kwargs['ocr_path'])
            self.ocr_path.mkdir(exist_ok=True)
        if 'force_ocr' in kwargs and kwargs['force_ocr']:
            self.force_ocr = kwargs['force_ocr']
        version = kwargs['version']
        if version:
            super(XDocPerturbationScore, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'],
                                                        version=version)
        else:
            super(XDocPerturbationScore, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'])
        self.BUILDER_CONFIGS = [XConfig(name=f"x{kwargs['doc_type']}.{lang}", lang=lang) for lang in _LANG]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "label": datasets.Value("uint8"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.pred_only:
            if self.is_tar_file:
                eval_file = self.data_dir / "eval" / "eval.tar.gz"
                train_file = self.data_dir / "train" / "train.tar.gz"
            else:
                eval_file = walk_dir(self.data_dir.as_posix())
                train_file = []
        else:
            if self.is_tar_file:
                eval_file = self.data_dir / "eval" / "eval.tar.gz"
                train_file = self.data_dir / "train" / "train.tar.gz"
            else:
                train_file = walk_dir(self.data_dir / "train")
                eval_file = walk_dir(self.data_dir / "eval")

        data_dir = dl_manager.download_and_extract({
            "train": train_file.as_posix(), "eval": eval_file.as_posix()}
        )
        if self.output_dir:
            train_sample = self.output_dir / "train.csv"
            eval_sample = self.output_dir / "eval.csv"
            if train_sample.exists():
                os.remove(train_sample)
            if eval_sample.exists():
                os.remove(eval_sample)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_or_paths": data_dir["train"],
                    "split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_or_paths": data_dir["eval"],
                    "split": "eval"}
            ),
        ]

    def _generate_examples(self, path_or_paths, split):
        """ Yields examples.
        这个方法将接收在前面的' _split_generators '方法中定义的' gen_kwargs '作为参数。
        It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        The key is not important, it's more here for legacy reason (legacy from tfds)
        它负责打开给定的文件并从数据集生成元组(键，示例)
        key是不重要的，更多的是为了传承

        这里就是根据自己的数据集来整理
        """
        max_samples = self.max_train_samples if split == 'train' else self.max_eval_samples
        file_dict = get_file_index(path_or_paths)

        if self.ocr_path and not self.is_tar_file:
            update_ocr_index(file_dict, self.ocr_path)

        keys = list(file_dict.keys())
        indices = np.arange(len(keys))
        np.random.shuffle(indices)

        select_keys = [keys[i] for i in indices[:max_samples]]
        for key in select_keys:
            file_group = file_dict[key]
            if 'img' not in file_group:
                print(f"Can't find img file of {key}")
                continue
            img_path = file_group['img']
            if 'ocr' not in file_group:
                if self.force_ocr:
                    ocr_data = ocr(img_path)
                    ocr_file = self.ocr_path / f"{key}.json"
                    with ocr_file.open("w") as f_ocr:
                        json.dump(ocr_data, f_ocr)
                else:
                    print(f"Can't find ocr file of {key}")
                    continue
            else:
                ocr_data = load_json(file_group['ocr'])
            if ocr_data is None:
                print("Skip ", key, ",", file_group)
                continue
            image, image_shape = load_image(img_path)

            lines = get_lines(ocr_data)
            for i, line in enumerate(lines):

                dummy_inputs, dummy_bbox, dummy_labels = get_sent_perturbation_word_level(self.tokenizer, line,
                                                                                          self.multiple_of_neg_samples,
                                                                                          image_shape)
                for j, (input_ids, bbox, label) in enumerate(zip(dummy_inputs, dummy_bbox, dummy_labels)):
                    guid = f"{key}-{i}-{j}"
                    yield guid, {
                        "id": guid,
                        "input_ids": input_ids,
                        "bbox": bbox,
                        "label": label,
                        "image": image,
                    }
