# -*- coding: utf-8 -*-
import os
import re


def do_predict(label_list, test_dataset, id_to_word, train_dataset, true_predictions):
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


def error_analysis(label_map, test_dataset, id_to_word, predictions):
    input_ids = test_dataset['input_ids']
    error_label_num = 0
    error_entity_num = 0
    miss_pred_entity = 0
    non_entity_num = 0

    pred_entity_write = []
    true_entity_write = []
    for i, (pred_labels, true_label_ids) in enumerate(zip(predictions, test_dataset['labels'])):
        true_labels = [label_map[label_id] for label_id in true_label_ids]
        pred_entity_span = parse_entity_span(input_ids[i], id_to_word, pred_labels)
        true_entity_span = parse_entity_span(input_ids[i], id_to_word, true_labels)
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
                entity_tokens.append(id_to_word[tokens[j]])
                j += 1
            entity = "".join(entity_tokens)
            entity_span[begin] = (curr_label_name, entity)
            entity_tokens.clear()
        else:
            j += 1
    return entity_span
