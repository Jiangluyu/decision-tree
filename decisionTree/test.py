from trees import *
from plot_tree import *


if __name__ == '__main__':
    data_set, labels = create_dataset()
    my_tree = create_tree(data_set, labels)
    print(my_tree)

    test_dataset = [[1, 0],
                    [1, 1],
                    [0, 0]]
    test_labels = ['no surfacing', 'flippers']

    for vec in test_dataset:
        test_result = classify(my_tree, test_labels, vec)
        print(test_result)

    createPlot(my_tree)

