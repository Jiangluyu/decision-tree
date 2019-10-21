from math import log
import operator


def calc_shannon_entropy(dataset):
    num_entries = len(dataset)
    label_counts = {}

    for feature_vec in dataset:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    shannon_ent = 0.0

    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent


def split_dataset(dataset, axis, value):
    ret_dataset = []

    for feature_vec in dataset:
        if feature_vec[axis] == value:
            reduced_vec = feature_vec[0:axis]
            reduced_vec.extend(feature_vec[axis+1:])
            ret_dataset.append(reduced_vec)

    return ret_dataset


def best_feature_to_split(dataset):
    num_features = len(dataset[0]) - 1
    best_ent = calc_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        features = [sample[i] for sample in dataset]
        unique_values = set(features)
        new_ent = 0.0

        for value in unique_values:
            sub_dataset = split_dataset(dataset, i, value)
            prob = float(len(sub_dataset)) / float(len(dataset))
            new_ent += prob * calc_shannon_entropy(sub_dataset)

        info_gain = best_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_count(class_list):
    class_count = {}

    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [sample[-1] for sample in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    best_feature = best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del(labels[best_feature])
    feature_values = [sample[best_feature] for sample in dataset]
    unique_values = set(feature_values)

    for value in unique_values:
        sub_labels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels)

    return my_tree


def create_dataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def classify(input_tree, feature_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    send_dict = input_tree[first_str]

    feature_index = feature_labels.index(first_str)
    class_label = None
    for key in send_dict.keys():
        if test_vec[feature_index] == key:
            if type(send_dict[key]).__name__ == 'dict':
                class_label = classify(send_dict[key], feature_labels, test_vec)
            else:
                class_label = send_dict[key]

    return class_label


