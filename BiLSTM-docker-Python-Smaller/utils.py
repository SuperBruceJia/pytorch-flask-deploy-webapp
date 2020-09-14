#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy


def format_result(result: list, text: str, tag: str) -> list:
    """
    Format the predicted results
    """
    entities = []
    for i in result:
        # If this is a single entity
        if len(i) == 1:
            single = i
            entities.append({
                "start": single[0],
                "stop": single[0] + 1,
                "word": text[single[0]:single[0]+1],
                "type": tag
            })
        else:
            # If this is not a single entity
            begin, end = i
            entities.append({
                "start": begin,
                "stop": end + 1,
                "word": text[begin:end+1],
                "type": tag
            })
    return entities


def get_tags(path: list, tag: str, tag_map: dict) -> list:
    """
    Get all the tags from the input sentence (index) results
    """
    begin_tag = tag_map.get("B-" + tag)
    mid_tag = tag_map.get("I-" + tag)
    end_tag = tag_map.get("E-" + tag)
    single_tag = tag_map.get("S-" + tag)
    o_tag = tag_map.get("O")

    begin = -1
    tags = []
    last_tag = 0

    for index, tag in enumerate(path):
        if tag == begin_tag:
            begin = index
        elif tag == end_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end])
        elif tag == single_tag:
            single = index
            tags.append([single])
        elif tag == o_tag:
            begin = -1
        last_tag = tag
    return tags


def tag_f1(tar_path: list, pre_path: list, tag: list, tag_map: dict) -> float:
    """
    Compute the Precision, Recall, and F1 Score
    """
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(path=tar, tag=tag, tag_map=tag_map)
        pre_tags = get_tags(path=pre, tag=tag, tag_map=tag_map)

        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print("\t{}\trecall {:.4f}\tprecision {:.4f}\tf1 {:.4f}".format(tag, recall, precision, f1))
    return recall, precision, f1


def tag_micro_f1(tar_path: list, pre_path: list, tags: list, tag_map: dict) -> float:
    """
    Compute the Precision, Recall, and F1 Score
    """
    origin = 0.
    found = 0.
    right = 0.

    # Iterate over different tags
    for tag in tags:
        # For this tag, accumulate the right scores
        for fetch in zip(tar_path, pre_path):
            tar, pre = fetch
            tar_tags = get_tags(path=tar, tag=tag, tag_map=tag_map)
            pre_tags = get_tags(path=pre, tag=tag, tag_map=tag_map)

            origin += len(tar_tags)
            found += len(pre_tags)

            for p_tag in pre_tags:
                if p_tag in tar_tags:
                    right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    return recall, precision, f1


def entity_label_f1(tar_path: list, pre_path: list, length: list, tag: str, tag_map: dict, prefix: str):
    """
    Compute the Precision, Recall, and F1 Score
    """
    c_tar_path = copy.deepcopy(tar_path)
    c_pre_path = copy.deepcopy(pre_path)

    origin = 0.
    found = 0.
    right = 0.

    new_tar = []
    new_pre = []
    true_pre = []

    prefix_tag = prefix + "-" + tag
    for fetch in zip(c_tar_path, c_pre_path, length):
        tar, pre, leng = fetch
        tar = tar[:leng]
        pre = pre[:leng]

        for index in range(len(tar)):
            prefix_tar = list(tag_map.keys())[list(tag_map.values()).index(tar[index])]
            if prefix_tar == prefix_tag:
                new_tar.append(tar[index])
                new_pre.append(pre[index])

            # Calculate the true number of the prediction
            prefix_pre = list(tag_map.keys())[list(tag_map.values()).index(pre[index])]
            if prefix_pre == prefix_tag:
                true_pre.append(tar[index])

    origin += len(new_tar)
    found += len(true_pre)

    for i in range(len(new_tar)):
        if new_tar[i] == new_pre[i]:
            right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print("\t{}\t{}\trecall {:.4f}\tprecision {:.4f}\tf1 {:.4f}".format(tag, prefix, recall, precision, f1))
    return recall, precision, f1


def label_f1(tar_path: list, pre_path: list, length: list, tags: list, tag_map: dict, prefix: str):
    """
    Compute the Precision, Recall, and F1 Score
    """
    c_tar_path = copy.deepcopy(tar_path)
    c_pre_path = copy.deepcopy(pre_path)

    origin = 0.
    found = 0.
    right = 0.

    new_tar = []
    new_pre = []
    true_pre = []

    # Iterate over tags
    for tag in tags:
        if prefix != "O":
            prefix_tag = prefix + "-" + tag
        else:
            prefix_tag = "O"

        for fetch in zip(c_tar_path, c_pre_path, length):
            tar, pre, leng = fetch
            tar = tar[:leng]
            pre = pre[:leng]

            for index in range(len(tar)):
                prefix_tar = list(tag_map.keys())[list(tag_map.values()).index(tar[index])]
                if prefix_tar == prefix_tag:
                    new_tar.append(tar[index])
                    new_pre.append(pre[index])

                # Calculate the true number of the prediction
                prefix_pre = list(tag_map.keys())[list(tag_map.values()).index(pre[index])]
                if prefix_pre == prefix_tag:
                    true_pre.append(tar[index])

        origin += len(new_tar)
        found += len(true_pre)

        for i in range(len(new_tar)):
            if new_tar[i] == new_pre[i]:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print("\t{}\trecall {:.4f}\tprecision {:.4f}\tf1 {:.4f}".format(prefix, recall, precision, f1))
    return recall, precision, f1
