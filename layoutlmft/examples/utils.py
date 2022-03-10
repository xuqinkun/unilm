# -*- coding: utf-8 -*-
import os
import re


def do_predict(label_list, test_dataset, tokenizer, train_dataset, training_args, true_predictions):
    id_to_word = {v: k for k, v in tokenizer.vocab.items()}
    # Save predictions
    # output_dir = os.path.join(training_args.output_dir, "predict",
    #                           "%d-%d" % (len(train_dataset), len(test_dataset)))
    pred_cloze_map = {}
    true_cloze_map = {}
    for i, _src in enumerate(test_dataset['id']):
        img_src, chunk_id = _src.rsplit("_", 1)
        _, file_dir = img_src.rsplit("/", 1)
        filename, suffix = file_dir.rsplit(".", 1)
        if "合同" in filename:
            key = filename.rsplit("_", 1)[0]
        else:
            key = filename
        if key not in pred_cloze_map:
            pred_cloze_map[key] = {}
            true_cloze_map[key] = {}
        true_labels = [label_list[i] for i in test_dataset['labels'][i]]
        pred_labels = true_predictions[i]
        input_ids = test_dataset['input_ids'][i]
        true_label_entity_pair = parse_key_value(input_ids, true_labels, id_to_word)
        pred_label_entity_pair = parse_key_value(input_ids, pred_labels, id_to_word)
        pred_cloze_map[key].update(pred_label_entity_pair)
        true_cloze_map[key].update(true_label_entity_pair)

    entire_doc_num = count_entire_doc(pred_cloze_map, true_cloze_map)
    total_label_num, correct_label_num, pred_label_num = count_correct_label(pred_cloze_map, true_cloze_map)
    total_doc_num = len(pred_cloze_map)

    # 计算recall，precision，f1
    recall = correct_label_num * 100 / total_label_num
    precision = correct_label_num * 100 / pred_label_num
    f1 = 2 * precision * recall / (precision + recall)

    train_pages = count_data_size(train_dataset)
    test_pages = count_data_size(test_dataset)

    accurate = entire_doc_num / total_doc_num
    print('entire_filed_doc=%d, total_doc=%d percent=%.2f%%' % (
        entire_doc_num, total_doc_num, 100 * accurate
    ))
    print("Train num=%d, Test num=%d" % (train_pages, test_pages))
    print("Total labels: %d Correct pred:%d pred_label: %d" %
          (total_label_num, correct_label_num, pred_label_num))
    print("Recall: %.2f%%" % recall)
    print("Precision: %.2f%%" % precision)
    print("F1: %.2f%%" % f1)


def write_ret(img_src, output_dir, key, filename):
    with open(img_src, "rb") as f:
        data = f.read()
    output_path = os.path.join(output_dir, key)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, filename), 'wb') as f:
        f.write(data)


def count_entire_doc(pred_cloze_map, true_cloze_map):
    cnt = 0
    for key, true_pairs in true_cloze_map.items():
        correct_filed_num = 0
        if key not in pred_cloze_map:
            continue
        pred_pairs = pred_cloze_map[key]
        pred_labels = set([k.split("-")[-1] for k, _ in pred_pairs.items() if k != 'O'])
        true_labels = set([k.split("-")[-1] for k, _ in true_pairs.items() if k != 'O'])

        for k, v in true_pairs.items():
            if k == 'O' or k not in pred_pairs:
                continue
            if pred_pairs[k] == v:
                correct_filed_num += 1
        if correct_filed_num == len(true_labels) and len(pred_labels) == len(true_labels):
            cnt += 1
    return cnt


def count_correct_label(pred_cloze_map, true_cloze_map):
    correct_label_num = 0
    total_label_num = 0
    pred_label_num = 0
    for key, true_item in true_cloze_map.items():
        for k, v in true_item.items():
            if k != 'O':
                total_label_num += 1
        if key not in pred_cloze_map:
            continue
        pred_filed = pred_cloze_map[key]
        for k, v in pred_filed.items():
            if k != 'O':
                pred_label_num += 1
        for k, v in true_item.items():
            if k == 'O' or k not in pred_filed:
                continue
            if pred_filed[k] == v:
                correct_label_num += 1
    return total_label_num, correct_label_num, pred_label_num


def parse_key_value(input_entity_id_list, label_list, id_to_word):
    idx = 0
    label_entity_pair = {}
    while idx < len(input_entity_id_list):
        if label_list[idx].startswith("B"):
            true_label = label_list[idx].split("-", 1)[-1]
            tmp_entity = []
            while idx < len(input_entity_id_list) and label_list[idx].split("-", 1)[-1] == true_label:
                word_id = input_entity_id_list[idx]
                idx += 1
                if word_id != 6:
                    word = id_to_word[word_id]
                    tmp_entity.append(word)
            if len(tmp_entity) != 0:
                entity = "".join(tmp_entity)
                if '▁' in entity:
                    candidate_entities = re.split('[▁:]', entity)
                    for entity in candidate_entities:
                        if entity.strip(' ') != '':
                            entity = entity.strip(' ')
                            break
                label_entity_pair[true_label] = entity
        idx += 1
    return label_entity_pair


def count_data_size(dataset):
    buffer = set()
    for i, _src in enumerate(dataset['id']):
        filename = _src.rsplit("/", 1)[-1]
        filename, chunk_id = filename.rsplit("_", 1)
        buffer.add(filename)
    return len(buffer)
