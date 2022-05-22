# coding=utf-8
from ast import literal_eval
import os
from nltk import word_tokenize
import datasets
from pathlib import Path
from layoutlm.deprecated.examples.seq_labeling.data.perturbations import get_local_neighbors_word_level, \
    get_local_neighbors_char_level
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


def get_word_label(root_dir: Path, file_index: dict, split):
    parent_dir = root_dir / split
    img_dir = parent_dir / "images"
    tag_dir = parent_dir / "tagged"
    img_files = sorted(img_dir.glob("*.jpg"))
    tag_files = sorted(tag_dir.glob("*.json"))
    remove_keys = []
    for img, tag in zip(img_files, tag_files):
        key = tag.stem
        assert img.stem == key
        if key not in file_index:
            remove_keys.append(key)
            continue
        item = file_index[key]
        if 'ocr' not in item:
            remove_keys.append(key)
            continue
        ocr_data = Path(item['ocr']).read_text()
        words_tag = read_json_file(tag)
        if ocr_data[-1] == '\n':
            ocr_data = ocr_data[:-1]
        lines = ocr_data.split("\n")
        words = words_tag['words']
        start = 0
        line_spans = []
        for line in lines:
            tokens = line.split(",")
            line_text = ",".join(tokens[8:])
            slices = word_tokenize(line_text)
            offset = 0
            # 文本可能出现部分字符不一致的情况，如果左右两个token都出现在文本行当中，
            # 那么中间的token也应该出现在当前行中
            while (start + offset < len(words) and offset < len(slices) and words[start + offset].lower() == slices[offset].lower()) \
                    or (start + offset + 1 < len(words) and offset + 1 < len(slices) and
                        words[start + offset + 1].lower() == slices[offset + 1].lower()):
                offset += 1
            line_spans.append((start, start + offset))
            start += offset
        for span, line in zip(line_spans, lines):
            words_line = " ".join(words[span[0]: span[1]])
            if words_line != line:
                print((words_line, "\t", line))
    print(len(remove_keys))


if __name__ == '__main__':
    root_dir = Path("/home/std2020/xuqinkun/data/sroie")
    sroie_words_dir = root_dir / 'sroie_words'
    sroie_raw_dir = root_dir / 'sroie_raw'
    file_index = get_file_index(sroie_raw_dir, 'processed')
    get_word_label(sroie_words_dir, file_index, 'train')
