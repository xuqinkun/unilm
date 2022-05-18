# -*- coding: utf-8 -*-
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ..data.ITMDataset import ITMDataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    WEIGHTS_NAME,
)
from transformers.utils import logging as logger

from .post_processing import convert_predictions_to_dict


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = ITMDataset(args, tokenizer, labels, pad_token_label_id, mode=mode)
    data_dir = Path(args.data_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )

    # Eval!
    print("***** Running evaluation %s *****" % prefix)
    print("  Num examples = %d" % len(eval_dataset))
    print("  Batch size = %d" % args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    files = []
    perturb_text_prob = 0.2
    mask_bbox_prob = 0.1
    left_bbox_prob = 0.2
    right_bbox_prob = 0.3
    narrow_bbox_prob = 0.4
    amplify_bbox_prob = 0.5

    print(f" perturb_text_prob = {perturb_text_prob}")
    print(f" mask_bbox_prob = {mask_bbox_prob}")
    print(f" left_bbox_prob = {left_bbox_prob}")
    print(f" right_bbox_prob = {right_bbox_prob}")
    print(f" narrow_bbox_prob = {narrow_bbox_prob}")
    print(f" amplify_bbox_prob = {amplify_bbox_prob}")

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch[0]
            new_ids = torch.randint_like(input_ids, tokenizer.vocab_size)
            prob_mat = torch.rand(input_ids.shape)
            input_ids = torch.where(prob_mat < perturb_text_prob, new_ids, input_ids)
            inputs = {
                "input_ids": input_ids.to(args.device),
                "attention_mask": batch[1].to(args.device),
                "labels": batch[3].to(args.device),
            }
            if args.model_type in ["layoutlm"]:
                perturb_bbox = torch.clone(batch[4])
                max_x = max(perturb_bbox[:, :, 0].max(), perturb_bbox[:, :, 2].max())
                max_y = max(perturb_bbox[:, :, 1].max(), perturb_bbox[:, :, 3].max())
                zero_bbox = torch.zeros_like(perturb_bbox)

                B, L, D = perturb_bbox.shape
                left_bbox = torch.cat((perturb_bbox[:, :-1, :], torch.zeros((B, 1, D), dtype=torch.int64)), dim=1)
                right_bbox = torch.cat((perturb_bbox[:, 1:, :], torch.zeros((B, 1, D), dtype=torch.int64)), dim=1)

                scale_bbox = torch.randint(0, 10, (B, L, 2))
                narrow_bbox = torch.cat((scale_bbox, -scale_bbox), dim=2)
                amplify_bbox = torch.cat((-scale_bbox, scale_bbox), dim=2)
                prob_mat = torch.rand((B, L)).unsqueeze(dim=2)
                prob_stack = prob_mat.repeat(1, 1, D)
                # Mask perturb_bbox
                perturb_bbox = torch.where(prob_stack < mask_bbox_prob, zero_bbox, perturb_bbox)
                # Replace by left perturb_bbox
                left_prob_mat = torch.where(prob_stack > left_bbox_prob, torch.zeros_like(prob_stack), prob_stack)
                perturb_bbox = torch.where(mask_bbox_prob <= left_prob_mat, left_bbox, perturb_bbox)
                # Replace by right perturb_bbox
                right_prob_mat = torch.where(prob_stack > right_bbox_prob, torch.zeros_like(prob_stack), prob_stack)
                perturb_bbox = torch.where(left_bbox_prob <= right_prob_mat, right_bbox, perturb_bbox)
                # Narrow perturb_bbox
                narrow_prob_mat = torch.where(prob_stack > narrow_bbox_prob, torch.zeros_like(prob_stack),
                                              prob_stack)
                perturb_bbox = torch.where(right_bbox_prob <= narrow_prob_mat, narrow_bbox, perturb_bbox)
                # Amplify perturb_bbox
                amplify_prob_mat = torch.where(prob_stack > amplify_bbox_prob, torch.zeros_like(prob_stack),
                                               prob_stack)
                perturb_bbox = torch.where(narrow_bbox_prob <= amplify_prob_mat, amplify_bbox, perturb_bbox)
                perturb_bbox = torch.where(perturb_bbox < 0, zero_bbox, perturb_bbox)
                ones_bbox = torch.ones((B, L), dtype=torch.int64)
                # Replace items which are greater than max
                perturb_bbox[:, :, 0] = torch.where(perturb_bbox[:, :, 0] > max_x, ones_bbox * max_x,
                                                    perturb_bbox[:, :, 0])
                perturb_bbox[:, :, 1] = torch.where(perturb_bbox[:, :, 1] > max_y, ones_bbox * max_y,
                                                    perturb_bbox[:, :, 1])
                perturb_bbox[:, :, 2] = torch.where(perturb_bbox[:, :, 2] > max_x, ones_bbox * max_x,
                                                    perturb_bbox[:, :, 2])
                perturb_bbox[:, :, 3] = torch.where(perturb_bbox[:, :, 3] > max_y, ones_bbox * max_y,
                                                    perturb_bbox[:, :, 3])
                left_upper_points = perturb_bbox[:, :, :2]
                right_bottom_points = perturb_bbox[:, :, 2:]
                left_upper_bbox = torch.cat((left_upper_points, left_upper_points), dim=2)
                right_bottom_bbox = torch.cat((right_bottom_points, right_bottom_points), dim=2)
                perturb_bbox = torch.where(left_upper_bbox > right_bottom_bbox, left_upper_bbox, perturb_bbox)
                assert torch.all(perturb_bbox[:, :, 2:] >= perturb_bbox[:, :, :2])
                assert torch.all(perturb_bbox >= 0)
                if amplify_bbox_prob > 0:
                    inputs["bbox"] = perturb_bbox.to(args.device)
                else:
                    inputs["bbox"] = batch[4].to(args.device)
            elif args.model_type == 'layoutlm_itm':
                inputs["bbox"] = batch[4].to(args.device)
                inputs['image'] = batch[5].to(args.device)
            inputs["token_type_ids"] = (
                batch[2].to(args.device)
                if args.model_type in ["bert", "layoutlm", 'layoutlm_itm']
                else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating
            files += list(batch[5])
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
    probs, preds = torch.max(F.softmax(torch.tensor(preds), dim=2), dim=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_classes = [[] for _ in range(out_label_ids.shape[0])]
    flatten_label_ids = out_label_ids.reshape((-1,)).tolist()
    flatten_pred_ids = torch.flatten(preds).numpy().tolist()
    flatten_label_list = []
    flatten_pred_list = []
    for label_id, pred_id in zip(flatten_label_ids, flatten_pred_ids):
        if label_id != pad_token_label_id:
            flatten_label_list.append(label_map[label_id])
            flatten_pred_list.append(label_map[pred_id])
    all_texts = eval_dataset.all_text
    for i in range(out_label_ids.shape[0]):
        tokens = []
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                token = tokenizer.convert_ids_to_tokens(eval_dataset.all_input_ids[i][j].item())
                tokens.append(token)
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(preds[i][j].item())
                preds_classes[i].append(label_map[preds[i][j].item()])
    print("Data:" + data_dir.parent.stem)
    annotation_path = data_dir.parent / 'test' / 'entities'
    probs = probs.cpu().numpy().tolist()
    num_pred = 0
    num_correct = 0
    num_gt = 0
    results = {}
    if args.model_type == 'layoutlm':
        if args.eval_strict:
            print("****Strict eval model, which will compare entity****")
            for file, text, pred_one_doc, probs in zip(files, all_texts, preds_list, probs):
                pred_entity = convert_predictions_to_dict(label_map, text, pred_one_doc, probs)
                num_pred += len(pred_entity)

                with open(os.path.join(annotation_path, file + ".txt"), 'r') as f:
                    gt_entity = json.load(f)
                num_gt += len(gt_entity)
                for key, value in gt_entity.items():
                    if key in pred_entity and pred_entity[key] == value:
                        num_correct += 1
            precision = num_correct / num_pred
            recall = num_correct / num_gt
            results = {"precision": precision,
                       "recall": recall,
                       "f1": 2 * precision * recall / (precision + recall) if precision != 0 and recall != 0 else 0
                       }
        else:
            results = {
                "precision": precision_score(flatten_label_list, flatten_pred_list),
                "recall": recall_score(flatten_label_list, flatten_pred_list),
                "f1": f1_score(flatten_label_list, flatten_pred_list),
            }
        report = classification_report(flatten_label_list, flatten_pred_list)
        print(report)
    elif args.model_type == 'layoutlm_itm':
        nb_correct = sum([1 if l1 == l2 else 0 for l1, l2 in zip(flatten_label_list, flatten_pred_list)])
        results = {"accuracy": nb_correct/len(flatten_label_list)}
    print("***** Eval results %s *****")
    for key in sorted(results.keys()):
        print(f"{key} = {100 * results[key]:.2f}%")

    return results, preds_classes


def do_eval(args, tokenizer_class, model_class, labels, pad_token_label_id):
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case
    )
    results = {}
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(
                glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            )
        )
        logger.setLevel("warn")
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result, _ = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="test",
            prefix=global_step,
        )
        if global_step:
            result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        results.update(result)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))