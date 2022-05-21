# coding=utf-8
from ast import literal_eval
import os
import json
import datasets
from pathlib import Path
from layoutlm.deprecated.examples.seq_labeling.data.perturbations import get_local_neighbors_word_level, get_local_neighbors_char_level
from layoutlm.deprecated.examples.seq_labeling.data.util import *
from layoutlmft.data.utils import load_image, simplify_bbox, normalize_bbox

logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{2019,
   title={ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
   url={http://dx.doi.org/10.1109/ICDAR.2019.00244},
   DOI={10.1109/icdar.2019.00244},
   journal={2019 International Conference on Document Analysis and Recognition (ICDAR)},
   publisher={IEEE},
   author={Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
   year={2019},
   month={Sep}
}
"""
_DESCRIPTION = """\
https://arxiv.org/abs/2103.10213
"""
if "DEBUG" in os.environ:
    print(f"\n{os.path.dirname(__file__)}\n")


class SroieConfig(datasets.BuilderConfig):
    """BuilderConfig for SROIE"""

    def __init__(self, **kwargs):
        """BuilderConfig for SROIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SroieConfig, self).__init__(**kwargs)


class SROIE_For_ITA(datasets.GeneratorBasedBuilder):
    """
    Load dataset Image Text Alignment: image, text, bbox
    Covered or uncovered
    """
    BUILDER_CONFIGS = [
        SroieConfig(name="sroie", version=datasets.Version("0.0.0"), description="SROIE dataset"),
    ]

    def __init__(self, **kwargs):
        if 'version' in kwargs:
            super(SROIE_For_ITA, self).__init__(cache_dir=kwargs['cache_dir'],
                                                name=kwargs['name'],
                                                version=kwargs['version'])
        else:
            super(SROIE_For_ITA, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'])
        self.data_dir = Path(kwargs['data_dir'])
        self.tokenizer = kwargs['tokenizer']
        self.overwrite_cache = kwargs['overwrite_cache'] if 'overwrite_cache' in kwargs else False

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "bbox": datasets.Sequence(datasets.Value("int64")),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "label": datasets.features.ClassLabel(
                            names=["covered", "uncovered"]
                        ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://arxiv.org/abs/2103.10213",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        to_dir = "processed"

        dest_dir = self.data_dir / to_dir

        train_dir = dest_dir / "train"
        eval_dir = dest_dir / "eval"
        test_dir = dest_dir / "test"

        if not dest_dir.exists() or self.overwrite_cache:
            file_index = get_file_index(self.data_dir, to_dir)
            eval_files, test_files, train_files = split_files(file_index)
            copy_files(train_files, train_dir)
            copy_files(eval_files, eval_dir)
            copy_files(test_files, test_dir)
        else:
            train_files = load_cache(train_dir)
            eval_files = load_cache(eval_dir)
            test_files = load_cache(test_dir)
        # move files from the second URL together with files from the first one.
        # downloaded_file = dl_manager.download_and_extract(self.data_dir.as_posix())
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_files}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": eval_files}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_files}
            ),
        ]

    def _generate_examples(self, filepath):
        for doc_id, item in enumerate(filepath):
            img_file = Path(item['img'])
            ocr_file = Path(item['ocr'])
            ocr_data = ocr_file.read_text()
            image, size = load_image(img_file)
            if ocr_data.endswith("\n"):
                ocr_data = ocr_data[: -1]
            lines = ocr_data.split("\n")
            for line_id, line in enumerate(lines):
                tokens = line.split(",")
                bbox = normalize_bbox(literal_eval(",".join(tokens[:8])), size)
                text = ",".join(tokens[8:]).lower()
                broken_texts = list(get_local_neighbors_char_level(text, 5))
                broken_texts.append(text)
                labels = ["uncovered"] * len(broken_texts)
                labels.append("covered")
                for i, (line, label) in enumerate(zip(broken_texts, labels)):
                    guid = f'{doc_id}-{line_id}-{i}'
                    tokenized_inputs = self.tokenizer(line,
                                                      add_special_tokens=False,
                                                      return_offsets_mapping=True,
                                                      return_attention_mask=False, )
                    feature = {
                        "id": guid,
                        "input_ids": tokenized_inputs["input_ids"],
                        "bbox": bbox,
                        "image": image,
                        "label": label,
                    }
                    yield guid, feature
