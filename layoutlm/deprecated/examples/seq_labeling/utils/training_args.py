# -*- coding: utf-8 -*-
from dataclasses import dataclass, field

from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    AutoTokenizer,
    LayoutLMv2Config,
)

from layoutlm.deprecated.examples.seq_labeling.models.modeling_ITA import LayoutlmForImageTextMatching
from layoutlm.deprecated.layoutlm.modeling.layoutlm import (
    LayoutlmConfig,
    LayoutlmForTokenClassification
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
    "layoutlm_itm": (LayoutLMv2Config, LayoutlmForImageTextMatching, AutoTokenizer),
}


@dataclass
class TrainingArgs:
    data_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    model_type: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pre-trained model or shortcut name selected in the list: "}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    labels: str = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    config_name: str = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name."}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name."}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pre-trained models downloaded from s3."}
    )

    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than "
                          "this will be truncated, sequences shorter will be padded."}
    )

    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training"}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run eval on the test set."}
    )
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    per_gpu_train_batch_size: int = field(
        default=8,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    per_gpu_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    eval_all_checkpoints: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation during training at each logging step."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={"help": "Overwrite the content of the output directory."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    eval_strict: bool = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory."}
    )
    server_ip: str = field(
        default="",
        metadata={"help": "Overwrite the content of the output directory."}
    )
    server_port: str = field(
        default="",
        metadata={"help": "Overwrite the content of the output directory."}
    )

