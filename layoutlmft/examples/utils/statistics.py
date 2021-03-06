# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import torch

from layoutlmft.data.datasets.xdoc import LABEL_MAP

BOE = '▁'
COLON = ":"
WEIGHTS_NAME = "pytorch_model.bin"

pattern_map = {
    "AMOUNT": "((\d{1,9})(,?(\d{1,9}))*(\.\d{1,9})?)",
    'SIGN_DATE': "\d{2}/\d{2}/\d{2,4}|\d{2,4}-\d{2}-\d{2}|\d{2,4}年\d{2}月\d{2}日|\d{2,4}/\d*",
}


def load_model(checkpoint, model):
    state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME))
    model.load_state_dict(state_dict, strict=False)


def do_predict(label_list, datasets, id_to_word, true_predictions, full_doc=False):
    pred_cloze_map = {}
    true_cloze_map = {}
    test_dataset = datasets["validation"]
    train_dataset = datasets["train"]
    for doc_key, input_ids, pred_labels, labels in zip(test_dataset['id'], test_dataset['input_ids'],
                                                       true_predictions, test_dataset['labels']):
        fullname, chunk_id = doc_key.rsplit("_", 1)
        if full_doc:
            key = fullname.split("_")[0]
        else:
            key = fullname
        if key not in pred_cloze_map:
            pred_cloze_map[key] = {}
            true_cloze_map[key] = {}
        true_labels = [label_list[l] for l in labels]
        # 提取键值对
        true_label_entity_pair = parse_key_value(input_ids, true_labels, id_to_word)
        pred_label_entity_pair = parse_key_value(input_ids, pred_labels, id_to_word)
        # 如果需要提取完整的文档，则需要将每次获取的键值对追加到同一篇文档中
        _assemble(key, pred_cloze_map, pred_label_entity_pair)
        _assemble(key, true_cloze_map, true_label_entity_pair)

    entire_doc_num = count_entire_doc(pred_cloze_map, true_cloze_map)
    total_label_num, correct_label_num, pred_label_num = compute_metric(pred_cloze_map, true_cloze_map)
    total_doc_num = len(pred_cloze_map)

    # 计算recall，precision，f1
    if total_label_num != 0:
        recall = correct_label_num * 100 / total_label_num
        precision = correct_label_num * 100 / pred_label_num
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0

    train_pages = count_data_size(train_dataset)
    test_pages = count_data_size(test_dataset)

    accurate = entire_doc_num / total_doc_num
    print('entire_field_doc=%d, total_doc=%d percent=%.2f%%\n' % (
        entire_doc_num, total_doc_num, 100 * accurate
    ))
    print("Train num=%d, Test num=%d" % (train_pages, test_pages))
    print("Total labels: %d Correct pred:%d pred_label: %d" %
          (total_label_num, correct_label_num, pred_label_num))
    print("Recall: %.2f%%" % recall)
    print("Precision: %.2f%%" % precision)
    print("F1: %.2f%%" % f1)
    return pred_cloze_map


def _assemble(key, cloze_map, label_entity_pair):
    for k in label_entity_pair:
        if k not in cloze_map[key]:
            cloze_map[key][k] = set()
        cloze_map[key][k].update(label_entity_pair[k])


def write_ret(img_src, output_dir, key, filename):
    with open(img_src, "rb") as f:
        data = f.read()
    output_path = os.path.join(output_dir, key)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, filename), 'wb') as f:
        f.write(data)


def count_entire_doc(pred_cloze_map, true_cloze_map):
    """完整文档预测结果统计，即整个文档中的所有实体标注正确将计数+1"""
    cnt = 0
    for key, true_pairs in true_cloze_map.items():
        correct_field_num = 0
        if key not in pred_cloze_map:
            continue
        pred_pairs = pred_cloze_map[key]
        pred_labels = set([k.split("-")[-1] for k, _ in pred_pairs.items() if k != 'O'])
        true_labels = set([k.split("-")[-1] for k, _ in true_pairs.items() if k != 'O'])
        # 如果预测出得标签和真实标签不一致则跳过
        if pred_labels != true_labels:
            continue
        # 计算预测正确的实体数
        for k, v in true_pairs.items():
            if k == 'O' or k not in pred_pairs:
                continue
            if pred_pairs[k] == v:
                correct_field_num += 1
        # 预测正确的实体数等于真实标记的实体数，则计数+1
        if correct_field_num == len(true_labels):
            cnt += 1
    return cnt


def compute_metric(pred_cloze_map, true_cloze_map):
    true_total = 0      # 真实实体总数
    pred_correct = 0    # 预测正确的实体数
    pred_total = 0      # 预测出的实体数
    for key, true_item in true_cloze_map.items():
        for k, v in true_item.items():
            if k != 'O':
                true_total += 1
        if key not in pred_cloze_map:
            continue
        pred_fields = pred_cloze_map[key]
        # 统计预测实体数
        for k, v in pred_fields.items():
            if k != 'O':
                pred_total += 1
        for k, v in true_item.items():
            if k == 'O' or k not in pred_fields:
                continue
            if pred_fields[k] == v:
                pred_correct += 1
    return true_total, pred_correct, pred_total


def parse_key_value(input_entity_id_list, label_list, id_to_word):
    """提取键值对"""
    idx = 0
    label_entity_pair = {}
    while idx < len(input_entity_id_list):
        # 实体标签以B开头
        try:
            if label_list[idx].startswith("B"):
                true_label = label_list[idx].split("-", 1)[-1]
                tmp_entity = []
                while idx < len(input_entity_id_list) and label_list[idx].split("-", 1)[-1] == true_label:
                    word_id = input_entity_id_list[idx]
                    idx += 1
                    # 6代表实体起点
                    if word_id != 6:
                        word = id_to_word[word_id]
                        tmp_entity.append(word)
                if len(tmp_entity) != 0:
                    entity = "".join(tmp_entity)
                    if BOE in entity:
                        entity = entity.replace(BOE, " ")
                    if true_label in pattern_map:
                        # 使用正则过滤输出
                        pattern = pattern_map[true_label]
                        candidates = re.compile(pattern).findall(entity)
                        if len(candidates) != 0:
                            if isinstance(candidates[0], tuple):
                                entity = candidates[0][0]
                            else:
                                entity = candidates[0]
                    if COLON in entity:
                        entity = entity.split(COLON, 1)[-1]
                    if true_label not in label_entity_pair:
                        # 使用set去除重复值
                        label_entity_pair[true_label] = set()
                    label_entity_pair[true_label].add(entity)
            else:
                idx += 1
        except KeyError as e:
            raise e
    return label_entity_pair


def count_data_size(dataset):
    buffer = set()
    for i, _src in enumerate(dataset['id']):
        filename = _src.rsplit("/", 1)[-1]
        filename, chunk_id = filename.rsplit("_", 1)
        buffer.add(filename)
    return len(buffer)


def error_analysis(label_map, test_dataset, id_to_word, predictions):
    input_ids = test_dataset['input_ids']
    error_label_num = 0
    error_entity_num = 0
    miss_pred_entity = 0
    non_entity_num = 0

    pred_entity_write = []
    true_entity_write = []
    for input_id, pred_labels, true_label_ids in zip(input_ids, predictions, test_dataset['labels']):
        true_labels = [label_map[label_id] for label_id in true_label_ids]
        pred_entity_span = parse_entity_span(input_id, id_to_word, pred_labels)
        true_entity_span = parse_entity_span(input_id, id_to_word, true_labels)
        pred_entity_write.append(pred_entity_span)
        true_entity_write.append(true_entity_span)
        # 1. 根据真实标注来统计预测标注错误的数目
        for start_pos in true_entity_span:
            true_label, true_entity = true_entity_span[start_pos]
            if start_pos in pred_entity_span:
                pred_label, pred_entity = pred_entity_span[start_pos]
                # 标注错误
                if pred_label != true_label:
                    error_label_num += 1
                # 实体与标注的不一致
                elif pred_entity != true_entity:
                    error_entity_num += 1
            else:
                # 实体标注缺失
                miss_pred_entity += 1
        # 2. 将非实体预测为实体的数目
        for pos in pred_entity_span:
            if pos not in true_entity_span:
                non_entity_num += 1
    # with open("/tmp/pred.txt", 'w') as f:
    #     f.write("start_index\tlabel\tentity\n")
    #     for entity_map in pred_entity_write:
    #         for key, item in entity_map.items():
    #             f.write(f"{key}\t\t{item[0]}\t{item[1]}\n")
    # with open("/tmp/true.txt", 'w') as f:
    #     f.write("start_index\tlabel\tentity\n")
    #     for entity_map in true_entity_write:
    #         for key, item in entity_map.items():
    #             f.write(f"{key}\t\t{item[0]}\t{item[1]}\n")

    total_error_num = error_label_num + error_entity_num + miss_pred_entity + non_entity_num
    print("Statistics:")
    print(f"error_label_num:{error_label_num}\t{(error_label_num * 100 / total_error_num):.2f}%\n"
          f"error_entity_num:{error_entity_num}\t{(error_entity_num * 100 / total_error_num):.2f}%\n"
          f"miss_pred_entity:{miss_pred_entity}\t{(miss_pred_entity * 100 / total_error_num):.2f}%\n"
          f"non_entity_num:{non_entity_num}\t{(non_entity_num * 100 / total_error_num):.2f}%\n"
          f"total_error_num:{total_error_num}")
    pass


def parse_entity_span(tokens, id_to_word, labels):
    j = 0
    entity_span = {}
    while j < len(labels):
        curr_label_name = labels[j].split("-")[-1]
        entity_tokens = []
        if labels[j].startswith("B"):
            begin = j
            while j < len(labels) and labels[j] != 'O' and \
                    labels[j].split("-")[-1] == curr_label_name:
                try:
                    word = id_to_word[tokens[j]]
                    entity_tokens.append(word)
                except TypeError as e:
                    raise e
                except Exception as e:
                    raise e
                j += 1
            entity = "".join(entity_tokens)
            entity_span[begin] = (curr_label_name, entity)
            entity_tokens.clear()
        else:
            j += 1
    return entity_span


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _copy_file(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    with open(dst, "wb") as f:
        f.write(data)


def output_pred(doc_type, pred_cloze_map, data_dir, copy_src=False):
    data = []
    index = []
    columns = list(LABEL_MAP[doc_type].values())
    add_columns = ["LAST_DIR"]

    parent_dir, doc_type = data_dir.rsplit("/", 1)
    pred_dir = os.path.join(parent_dir, "pred", doc_type)
    _make_dir_if_not_exists(pred_dir)

    if copy_src:
        src_file = "eval.tar.gz"
        eval_src = os.path.join(data_dir, "eval", src_file)
        _copy_file(eval_src, os.path.join(pred_dir, src_file))
    for key in pred_cloze_map:
        if os.sep in key:
            last_dir, filename = key.rsplit(os.sep, 1)
            last_dir = last_dir.rsplit(os.sep, 1)[-1]
            index.append(filename)
        else:
            last_dir = ""
        field_map = pred_cloze_map[key]
        row_data = [last_dir]
        # 读取一行
        for column_name in columns:
            if column_name in field_map:
                row_data.append(";".join(field_map[column_name]))
            else:
                row_data.append("")
        data.append(row_data)
    df = pd.DataFrame(data=data, index=index, columns=add_columns+columns)
    output_path = os.path.join(pred_dir, "pred.csv")
    df.to_csv(output_path, encoding="utf_8_sig")
    print(output_path)