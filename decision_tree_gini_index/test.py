from trees import *
from plot_tree import *

if __name__ == '__main__':
    data_set, labels, labels_full = create_dataset()
    my_tree = create_tree(data_set, labels)
    print(my_tree)

    test_dataset = [['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.585, 0.002],
                    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.678, 0.370]]
    test_labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
    test_label_properties = [0, 0, 0, 0, 0, 0, 1, 1]
    for vec in test_dataset:
        test_result = classify(my_tree, test_labels, test_label_properties, vec)
        print(test_result)

    createPlot(my_tree)

