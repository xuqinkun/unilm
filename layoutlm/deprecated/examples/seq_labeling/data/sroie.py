# coding=utf-8
import json
import os
from pathlib import Path

import datasets
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


class SroieConfig(datasets.BuilderConfig):
    """BuilderConfig for SROIE"""

    def __init__(self, **kwargs):
        """BuilderConfig for SROIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SroieConfig, self).__init__(**kwargs)


class Sroie(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SroieConfig(name="sroie", version=datasets.Version("1.0.0"), description="SROIE dataset"),
    ]

    def __init__(self, **kwargs):
        super(Sroie, self).__init__(cache_dir=kwargs['cache_dir'], name=kwargs['name'])
        self.data_dir = Path(kwargs['data_dir']) / 'sroie.zip'
        self.tokenizer = kwargs['tokenizer']

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS",
                                   "B-TOTAL", "I-TOTAL"]
                        )
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
        downloaded_file = dl_manager.download_and_extract(self.data_dir.as_posix())
        # move files from the second URL together with files from the first one.
        dest = Path(downloaded_file) / "sroie"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest / "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest / "test"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "tagged")
        img_dir = os.path.join(filepath, "images")
        for doc_id, fname in enumerate(sorted(os.listdir(img_dir))):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, fname)
            image, size = load_image(image_path)
            boxes = [normalize_bbox(simplify_bbox(box), size) for box in data["bbox"]]
            words, labels = data["words"], data["labels"]
            for i, (word, label, box) in enumerate(zip(words, labels, boxes)):
                ids = self.tokenizer(word, add_special_tokens=False,)['input_ids']
                boxes = [box] * len(ids)
                if label.startswith('B'):
                    _, label_name = label.split('-')
                    labels = [f'I-{label_name}'] * len(ids)
                    labels[0] = f'B-{label_name}'
                else:
                    labels = [label] * len(ids)
                guid = f'{doc_id}-{i}'
                yield guid, {"id": str(guid),
                             "words": data["words"],
                             "input_ids": ids,
                             "bboxes": boxes,
                             "ner_tags": labels,
                             "image": image,
                             }
