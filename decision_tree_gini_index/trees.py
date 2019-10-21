import collections
from math import pow
import operator


def calc_gini_index(dataset):
    num_entries = len(dataset)
    label_counts = collections.defaultdict(int)

    for feature_vec in dataset:
        current_label = feature_vec[-1]
        label_counts[current_label] += 1

    gini_index = 1.0

    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        gini_index -= pow(prob, 2)

    return gini_index


def split_dataset_for_series(dataset, axis, value):
    elt_dataset = []
    gt_dataset = []

    for feature in dataset:
        if feature[axis] <= value:
            elt_dataset.append(feature)
        else:
            gt_dataset.append(feature)

    return elt_dataset, gt_dataset


def split_dataset(dataset, axis, value):
    ret_dataset = []

    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_vec = feature_vec[:axis]
            reduced_vec.extend(feature_vec[axis + 1:])
            ret_dataset.append(reduced_vec)

    return ret_dataset


def calc_gini_gain_for_series(dataset, i, base_gini):
    best_delta_gini = 0.0
    best_split_point = -1

    feature_list = [example[i] for example in dataset]
    class_list = [example[-1] for example in dataset]
    dict_list = dict(zip(feature_list, class_list))

    sorted_feature_list = sorted(dict_list.items(), key=operator.itemgetter(0))
    num_feature_list = len(sorted_feature_list)
    split_point_list = [round((sorted_feature_list[i][0] + sorted_feature_list[i + 1][0]) / 2.0, 3) for i in
                        range(num_feature_list - 1)]

    # 计算出各个划分点信息增益
    for split_point in split_point_list:
        elt_dataset, gt_dataset = split_dataset_for_series(dataset, i, split_point)

        new_gini = len(elt_dataset) / len(sorted_feature_list) * calc_gini_index(elt_dataset) \
            + len(gt_dataset) / len(sorted_feature_list) * calc_gini_index(gt_dataset)

        delta_gini = base_gini - new_gini
        if delta_gini < best_delta_gini:
            best_split_point = split_point
            best_delta_gini = delta_gini

    return best_delta_gini, best_split_point


def calc_delta_gini(dataset, feature_list, i, base_gini):
    unique_values = set(feature_list)
    new_gini = 1.0

    for value in unique_values:
        sub_dataset = split_dataset(dataset=dataset, axis=i, value=value)
        prob = len(sub_dataset) / float(len(dataset))
        new_gini -= prob * calc_gini_index(sub_dataset)

    delta_gini = base_gini - new_gini

    return delta_gini


def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_gini = calc_gini_index(dataset)
    best_delta_gini = 1.0
    best_feature = -1
    flag_series = 0
    best_split_point = 0.0
    new_split_point = 0.0

    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        if isinstance(feature_list[0], str):
            delta_gini = calc_delta_gini(dataset, feature_list, i, base_gini)
        else:
            delta_gini, new_split_point = calc_gini_gain_for_series(dataset, i, base_gini)

        if delta_gini < best_delta_gini:
            best_delta_gini = delta_gini
            best_feature = i
            flag_series = 0

            if not isinstance(dataset[0][best_feature], str):
                flag_series = 1
                best_split_point = new_split_point

    if flag_series:
        return best_feature, best_split_point
    else:
        return best_feature


def create_dataset():
    dataset = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
               ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
               ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
               ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
               ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
               ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
               ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
               ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
               ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
               ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
               ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
               ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
               ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
               ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']]

    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

    labels_full = {}

    for i in range(len(labels)):
        label_list = [example[i] for example in dataset]
        unique_label = set(label_list)
        labels_full[labels[i]] = unique_label

    return dataset, labels, labels_full


def majority_count(class_list):
    class_count = collections.defaultdict(int)

    for vote in class_list:
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]

    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(dataset[0]) == 1:
        return majority_count(class_list)

    best_feature = choose_best_feature_to_split(dataset=dataset)

    split_point = 0.0

    # 如果是元组的话，说明此时是连续值
    if isinstance(best_feature, tuple):
        best_feature_label = str(labels[best_feature[0]]) + '<' + str(best_feature[1])
        split_point = best_feature[1]
        best_feature = best_feature[0]
        flag_series = 1
    else:
        best_feature_label = labels[best_feature]
        flag_series = 0

    my_tree = {best_feature_label: {}}

    feature_values = [example[best_feature] for example in dataset]

    # 连续值处理
    if flag_series:
        elt_dataset, gt_dataset = split_dataset_for_series(dataset, best_feature, split_point)
        sub_labels = labels[:]
        sub_tree = create_tree(elt_dataset, sub_labels)
        my_tree[best_feature_label]['<'] = sub_tree

        sub_tree = create_tree(gt_dataset, sub_labels)
        my_tree[best_feature_label]['>'] = sub_tree

        return my_tree

    # 离散值处理
    else:
        del (labels[best_feature])
        unique_values = set(feature_values)
        for value in unique_values:
            sub_labels = labels[:]
            sub_tree = create_tree(split_dataset(dataset=dataset, axis=best_feature, value=value), sub_labels)
            my_tree[best_feature_label][value] = sub_tree
        return my_tree


def classify(input_tree, feature_labels, feature_label_properties, test_vec):
    first_str = list(input_tree.keys())[0]
    first_label = first_str
    less_index = str(first_str).find('<')
    # print(less_index)
    if less_index > -1:  # 如果是连续型的特征
        first_label = str(first_str)[:less_index]
        # print("first_label", first_label)
    second_dict = input_tree[first_str]
    feature_index = feature_labels.index(first_label)  # 跟节点对应的特征
    # print(second_dict)
    class_label = None
    for key in second_dict.keys():  # 对每个分支循环
        if feature_label_properties[feature_index] == 0:  # 离散的特征
            if test_vec[feature_index] == key:  # 测试样本进入某个分支
                if type(second_dict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    class_label = classify(second_dict[key], feature_labels, feature_label_properties, test_vec)
                else:  # 如果是叶子， 返回结果
                    class_label = second_dict[key]
        else:
            split_point = float(str(first_str)[less_index + 1:])
            if test_vec[feature_index] < split_point:  # 进入左子树
                if type(second_dict['<']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    class_label = classify(second_dict['<'], feature_labels, feature_label_properties, test_vec)
                else:  # 如果是叶子， 返回结果
                    class_label = second_dict['<']
            else:
                if type(second_dict['>']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    class_label = classify(second_dict['>'], feature_labels, feature_label_properties, test_vec)
                else:  # 如果是叶子， 返回结果
                    class_label = second_dict['>']

    return class_label
