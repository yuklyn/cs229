import matplotlib.pyplot as plt
import numpy as np

import my_svm


def read_data(data_file):
    file_data = open(data_file, 'r')
    file_note = file_data.readline()
    print(file_note)
    sample_count, feature_count = [int(s) for s in file_data.readline().strip().split()]
    feature_names = file_data.readline().strip().split()
    data_set_x = np.zeros((sample_count, feature_count))
    data_set_y = np.zeros(sample_count)

    for sample_index, sample_data in enumerate(file_data):
        sample_values = [int(x) for x in sample_data.strip().split()]
        data_set_y[sample_index] = sample_values[0]
        feature_values = np.array(sample_values[1:])

        # 为什么对数据集这样处理？
        k = np.cumsum(feature_values[:-1:2])
        v = feature_values[1::2]

        data_set_x[sample_index, k] = v

    data_set_y = data_set_y * 2 - 1
    # 将 > 0 的元素全置为1
    data_set_x = 1. * (data_set_x > 0)

    return data_set_x, feature_names, data_set_y


def read_tow_dimension_matrix(data_file):
    file_data = open(data_file, 'r')
    file_note = file_data.readline()
    print(file_note)
    sample_count, feature_count = [int(s) for s in file_data.readline().strip().split()]
    feature_names = file_data.readline().strip().split()
    data_set_x = np.zeros((sample_count, feature_count))
    data_set_y = np.zeros(sample_count)

    for sample_index, sample_data in enumerate(file_data):
        sample_values = [float(x) for x in sample_data.strip().split()]
        data_set_y[sample_index] = sample_values[2]
        data_set_x[sample_index] = np.array(sample_values[0:2])

    return data_set_x, feature_names, data_set_y


def spam_classification_svm():
    import time
    start_time = time.time()
    print(start_time)
    errors = []

    # 对于线程核函数，epsilon和tolerance是要分开的；
    # 线性核的η会比较大，所有α会很小，更新值也很小，因此epsilon要小；
    # 但对于tolerance，线性核与非线性核代表的度量是一样的。

    # svm_my = my_svm.SVM(0.1, 0.000005, 100, my_svm.Linear)

    svm_my = my_svm.SVM(0.1, 0.01, 100, my_svm.RBF)

    # my_svm.DEBUG = False

    test_set_x, test_feature_names, test_set_y = read_data('data_set\\MATRIX.TEST')

    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.50')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))
    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.100')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))
    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.200')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))
    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.400')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))
    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.800')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))
    data_set_x, feature_names, data_set_y = read_data('data_set\\MATRIX.TRAIN.1400')
    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))

    duration = time.time() - start_time
    print(duration)

    x = [50, 100, 200, 400, 800, 1400]
    y = errors

    plt.title('SVM')
    plt.xlabel('Data Size')
    plt.ylabel('Error(%)')
    plt.plot(x, y)
    plt.show()

    return


def two_dimension_svm():
    errors = []

    # svm_my = my_svm.SVM(0.1, 0.01, 100, my_svm.Linear)

    svm_my = my_svm.SVM(0.00001, 0.00001, 100000, my_svm.RBF)

    import time
    start_time = time.time()
    print(start_time)

    data_set_x, feature_names, data_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN')
    test_set_x, test_feature_names, test_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN')
    # data_set_x, feature_names, data_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN.3')
    # test_set_x, test_feature_names, test_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN.3')
    # data_set_x, feature_names, data_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN.5')
    # test_set_x, test_feature_names, test_set_y = read_tow_dimension_matrix('data_set\\TWODIMENSION.TRAIN.5')

    svm_my.train(data_set_x, data_set_y)
    false_count = svm_my.test(test_set_x, test_set_y)
    errors.append(false_count * 100.0 / len(test_set_y))

    duration = time.time() - start_time
    print(duration)

    omega = svm_my.compute_omega()
    # normal_omega = np.sqrt(omega.dot(omega.T))
    print(omega)
    # output = svm_my.predict(test_set_x)

    # from matplotlib.patches import Circle
    # ax = plt.figure().add_subplot(111)
    # for index, data_x in enumerate(test_set_x):
    #     print(abs(output[0][index]) / normal_omega)
    #     circle = Circle(xy=(data_x[0], data_x[1]), radius=abs(output[0][index]) / normal_omega, alpha=0.4)
    #     ax.add_patch(circle)

    # plt.axis('scaled')
    # plt.axis('equal')

    x0 = 0
    y0 = (-svm_my.b - omega[0] * x0) / omega[1]
    x3 = 7
    y3 = (-svm_my.b - omega[0] * x3) / omega[1]

    plt.title("SVM")
    plt.xlim(xmax=7, xmin=0)
    plt.ylim(ymax=7, ymin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    x = data_set_x.T[0]
    y = data_set_x.T[1]
    for i, data_y in enumerate(data_set_y):
        if data_y == 1:
            plt.plot(x[i], y[i], 'r.')
        else:
            plt.plot(x[i], y[i], 'b.')
    plt.plot([x0, x3], [y0, y3])
    plt.show()

    return


if __name__ == '__main__':
    # two_dimension_svm()
    spam_classification_svm()
