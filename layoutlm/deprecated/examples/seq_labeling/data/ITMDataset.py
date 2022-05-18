# -*- coding: utf-8 -*-
import os
import sys
import logging
import torch
import random
from torch.utils.data import Dataset
from ast import literal_eval
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from layoutlmft.data.utils import load_image, normalize_bbox, simplify_bbox
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ITMDataset(Dataset):
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        # if args.local_rank not in [-1, 0] and mode == "train":
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        #     torch.distributed.barrier()

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            self.all_text = []
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(Path(args.data_dir), mode, tokenizer.vocab)
            features = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                local_rank=args.local_rank,
                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args.model_type in ["roberta"]),
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                pad_token_label_id=pad_token_label_id,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache
        # if args.local_rank == 0 and mode == "train":
        #     torch.distributed.barrier()

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_images = torch.stack([f.image for f in features])
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
            self.all_images[index],
        )


def read_examples_from_file(data_dir: Path, mode, vocab: dict):
    box_dir = data_dir / mode / 'box'
    image_dir = data_dir / mode / 'img'
    box_files = sorted(box_dir.glob("*.txt"))
    img_files = sorted(image_dir.glob("*.jpg"))
    guid_index = 1
    examples = []
    for img_file, box_file in tqdm(zip(img_files, box_files)):
        assert img_file.stem == box_file.stem
        content = box_file.read_text().split("\n")
        lines, labels, bboxes = [], [], []
        image, image_size = load_image(img_file)
        for line in content:
            if line.strip() == '':
                continue
            items = line.split(",")
            text = ",".join(items[8:])
            lines.append(text)
            box = literal_eval(",".join(items[:8]))
            bboxes.append(normalize_bbox(simplify_bbox(box), image_size))
            labels.append('covered')
        examples.append(
            InputExample(
                guid="{}-{}".format(mode, guid_index),
                lines=lines,
                labels=labels,
                bboxes=bboxes,
                image_size=image_size,
                image=image,
            )
        )

    perturb(examples, vocab)
    return examples


def perturb(examples: list, vocab: dict):
    # Perturbation for image text alignment and 15% uncovered,
    # among which 30% random line, 30% random image and 40% random bbox
    it_uncovered_prob = 0.15
    replace_text_prob = 0.8
    wrong_img_prob = 0.6
    replace_token_prob = 0.7
    replace_line_prob = 0.9
    sample_size = len(examples)
    id2tokens = {v: k for k, v in vocab.items()}
    for i, example in enumerate(examples):
        for j in range(len(example.lines)):
            if random.random() < it_uncovered_prob:
                continue
            # 85%概率为uncovered image text
            prob = random.random()
            example.labels[j] = 'uncovered'
            if prob < replace_text_prob:
                # 80%概率进行文本替换
                p = random.random()
                if p < replace_line_prob:
                    line = example.lines[j]
                    words = line.split(" ")
                    for k, w in enumerate(words):
                        if random.random() < replace_token_prob:
                            # replace_token_prob概率用同字典中的其他token进行替换
                            new_j = random.randint(0, len(vocab) - 1)
                            while j == new_j:
                                new_j = random.randint(0, len(vocab) - 1)
                            words[k] = id2tokens[new_j]
                    example.lines[j] = " ".join(words)
                else:
                    # 10%概率用其他文档文本替换
                    new_id = random.randint(0, sample_size - 1)
                    while new_id == i:
                        new_id = random.randint(0, sample_size - 1)
                    target_example = examples[new_id]
                    new_j = random.randint(0, len(target_example.lines) - 1)
                    while j == new_j:
                        new_j = random.randint(0, len(target_example.lines) - 1)
                    examples[i].lines[j] = target_example.lines[new_j]
            # elif prob < wrong_img_prob:
            #     # 10% 概率替换image
            #     example.image = examples[new_id].image
            else:
                # 剩下20%概率替换bbox
                new_j = random.randint(0, len(example.bboxes) - 1)
                while j == new_j:
                    new_j = random.randint(0, len(example.bboxes) - 1)
                example.bboxes[j] = example.bboxes[new_j]


class InputExample:

    def __init__(self, guid, lines, labels, bboxes, image, image_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.lines = lines
        self.labels = labels
        self.bboxes = bboxes
        self.image = image
        self.image_size = image_size


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        local_rank,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    flag = True
    for example in tqdm(examples):
        for box, label, line in zip(example.bboxes, example.labels, example.lines):

            word_tokens = tokenizer.tokenize(line)
            token_boxes = [box] * len(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids = [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(word_tokens) > max_seq_length - special_tokens_count:
                word_tokens = word_tokens[: (max_seq_length - special_tokens_count)]
                token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            word_tokens += [sep_token]
            token_boxes += [sep_token_box]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                word_tokens += [sep_token]
                token_boxes += [sep_token_box]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(word_tokens)

            if cls_token_at_end:
                word_tokens += [cls_token]
                token_boxes += [cls_token_box]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                word_tokens = [cls_token] + word_tokens
                token_boxes = [cls_token_box] + token_boxes
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(word_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = (
                                     [0 if mask_padding_with_zero else 1] * padding_length
                             ) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                token_boxes = ([pad_token_box] * padding_length) + token_boxes
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                token_boxes += [pad_token_box] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(token_boxes) == max_seq_length

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    boxes=token_boxes,
                    image=example.image,
                    image_size=example.image_size,
                )
            )
            if local_rank in [-1, 0] and flag:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in word_tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
                flag = False
    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            boxes,
            image,
            image_size,
    ):
        assert (
                0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.image = image
        self.image_size = image_size
