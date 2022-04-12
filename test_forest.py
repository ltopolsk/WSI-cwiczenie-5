from random_forest import RandomForest
from random import choice
import matplotlib.pyplot as plt


def import_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            read_line = line.rstrip('\n')
            point_to_add = []
            for item in read_line.split(','):
                point_to_add.append(float(item))
            data.append(point_to_add)
    return data


def cross_validation_split(dataset, n_folds):
    valid_set = []
    valid_len = int(len(dataset) / 4)
    dataset_copy = list(dataset)
    while len(valid_set) < valid_len:
        item_to_add = choice(dataset_copy)
        valid_set.append(item_to_add)
        dataset_copy.remove(item_to_add)
    dataset_split = []
    fold_size = int(len(dataset_copy) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            item_to_add = choice(dataset_copy)
            fold.append(item_to_add)
            dataset_copy.remove(item_to_add)
        dataset_split.append(fold)
    return dataset_split, valid_set


def evaluate_classifcation(dataset, n_folds, tree_max_depth=4, tree_min_size=10, amount=50, cat_index=0):
    folds, valid_set = cross_validation_split(dataset, n_folds)
    confussion_matrixes = []
    valid_matrixes = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        new_forest = RandomForest(train_set, tree_max_depth, tree_min_size, amount, cat_index)
        test_set = list(fold)
        confussion_matrixes.append(test_forest(new_forest, test_set, cat_index))
        valid_matrixes.append(test_forest(new_forest, valid_set, cat_index))
    return confussion_matrixes, valid_matrixes


def test_forest(forest, test_set, cat_index):
     """
        confussion matrix:
        predicted      actual
                    1    2    3
        1           p11  p12  p13
        2           p21  p22  p23
        3           p31  p32  p33

        Pmn - object m classified as n
    """
    confussion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for row in test_set:
        class_result = forest.classify_item(row)
        confussion_matrix[int(row[cat_index]) - 1][int(class_result) - 1] += 1
    return confussion_matrix


def get_cat_confusion_matrix(category, confusion_matrix):
    """
        tp fp
        fn tn
    """
    to_ret = [[], []]
    help_var = category - 1
    len_row_matrix = len(confusion_matrix[0])
    tp = confusion_matrix[help_var][help_var]
    tn = (confusion_matrix[(help_var - 1) % len_row_matrix][(help_var - 1) % len_row_matrix] +
          confusion_matrix[(help_var - 1) % len_row_matrix][(help_var + 1) % len_row_matrix] +
          confusion_matrix[(help_var + 1) % len_row_matrix][(help_var - 1) % len_row_matrix] +
          confusion_matrix[(help_var + 1) % len_row_matrix][(help_var + 1) % len_row_matrix])
    fp = (confusion_matrix[help_var][(help_var - 1) % len_row_matrix] +
          confusion_matrix[help_var][(help_var + 1) % len_row_matrix])
    fn = (confusion_matrix[(help_var - 1) % len_row_matrix][help_var] +
          confusion_matrix[(help_var + 1) % len_row_matrix][help_var])
    to_ret[0].append(tp)
    to_ret[0].append(fp)
    to_ret[1].append(fn)
    to_ret[1].append(tn)
    return to_ret


def get_accuracy(confusion_matrix):
    numerator = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    denominator = (confusion_matrix[0][1] + confusion_matrix[0][2] +
                   confusion_matrix[1][0] + confusion_matrix[1][2] + 
                   confusion_matrix[2][0] + confusion_matrix[2][1] + 
                   numerator)
    return round(numerator / denominator, 2)


def get_cat_precision(category, confusion_matrix):
    cat_matrix = get_cat_confusion_matrix(category, confusion_matrix)
    if cat_matrix[0][0] + cat_matrix[0][1] == 0:
        return 0
    else:
        return round(cat_matrix[0][0] / (cat_matrix[0][0] + cat_matrix[0][1]), 2)


def get_cat_sensitivity(category, confusion_matrix):
    cat_matrix = get_cat_confusion_matrix(category, confusion_matrix)
    if cat_matrix[0][0] + cat_matrix[1][0] == 0:
        return 0
    else:
        return round(cat_matrix[0][0] / (cat_matrix[0][0] + cat_matrix[1][0]), 2)


def get_cat_specificity(category, confusion_matrix):
    cat_matrix = get_cat_confusion_matrix(category, confusion_matrix)
    if cat_matrix[1][1] + cat_matrix[0][1] == 0:
        return 0
    else:
        return round(cat_matrix[1][1] / (cat_matrix[1][1] + cat_matrix[0][1]), 2)


def print_matrix(matrix):
    for line in matrix:
        print(line)


def get_plot(num_col_1, num_col_2, data):
    x = []
    for item in data:
        x.append(item[num_col_1])
    y = []
    for item in data:
        y.append(item[num_col_2])
    plt.xlabel(f"collumn {num_col_1 + 1}")
    plt.ylabel(f"collumn {num_col_2 + 1}")
    plt.scatter(x, y)

    plt.savefig(f"collumn_{num_col_1 + 1}.png", dpi = 72)
    plt.show()

if __name__ == "__main__":
    data = import_data("wine.data")
    """
    for i in range(1, len(data[0])):
        get_plot(i, 0, data)
    """
    matrixes, valid_matrixes = evaluate_classifcation(data, 3, 4, 10, 100)
    categories = list(set(int(row[0]) for row in data))
    mean_accuracy = 0
    i = 1
    for matrix in matrixes:
        print(f"Las {i}.:\nMacierz pomyłek: ")
        print_matrix(matrix)
        mean_accuracy += get_accuracy(matrix)
        print(f"Dokładność: {get_accuracy(matrix)}")
        for item in categories:
            print(f"precyzja kategorii {item}.: {get_cat_precision(item, matrix)}")
            print(f"czułość kategorii {item}.: {get_cat_sensitivity(item, matrix)}")
        i += 1
    print()
    mean_accuracy /= (i - 1)
    print(f"średnia dokładność wszystkich modeli: {round(mean_accuracy, 2)}")
    j = 1
    mean_accuracy = 0
    for matrix in valid_matrixes:
        print(f"Las {j}.:\nMacierz pomyłek: ")
        print_matrix(matrix)
        mean_accuracy += get_accuracy(matrix)
        print(f"Dokładność dla zbioru walidacyjnego: {get_accuracy(matrix)}")
        for item in categories:
            print(f"precyzja kategorii {item}.: {get_cat_precision(item, matrix)}")
            print(f"czułość kategorii {item}.: {get_cat_sensitivity(item, matrix)}")
        j += 1
    mean_accuracy /= (j - 1)
    print(f"średnia dokładność wszystkich modeli dla zbioru walidacyjnego: {round(mean_accuracy, 2)}")
