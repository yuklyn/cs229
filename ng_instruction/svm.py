import numpy as np

tau = 8.


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    category = (np.array(Y) * 2) - 1
    return matrix, tokens, category


def svm_train(train_set_x, train_set_y):
    state = {}

    sample_count, feature_count = train_set_x.shape

    train_set_x = 1. * (train_set_x > 0)

    squared = np.sum(train_set_x * train_set_x, axis=1)
    gram = train_set_x.dot(train_set_x.T)
    kernel_values = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)))
    # 将所有核函数值算了出来，得到对称矩阵

    alpha_set = np.zeros(sample_count)
    alpha_set_avg = np.zeros(sample_count)
    L = 1. / (64 * sample_count)
    outer_loops = 40

    for ii in range(outer_loops * sample_count):
        i = int(np.random.rand() * sample_count)
        margin = train_set_y[i] * np.dot(kernel_values[i, :], alpha_set)
        grad = sample_count * L * kernel_values[:, i] * alpha_set[i]
        if (margin < 1):
            grad -= train_set_y[i] * kernel_values[:, i]
        alpha_set -= grad / np.sqrt(ii + 1)
        alpha_set_avg += alpha_set

    alpha_set_avg /= (ii + 1) * sample_count

    state['alpha'] = alpha_set
    state['alpha_avg'] = alpha_set_avg
    state['Xtrain'] = train_set_x
    state['Sqtrain'] = squared
    ####################
    return state


def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error * 100


def main():
    errors = []
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.50')
    state = svm_train(trainMatrix, trainCategory)
    output = svm_test(testMatrix, state)

    fails = []
    for index, result in enumerate(output):
        if result != testCategory[index]:
            fails.append(index)
    print(fails)
    print(len(fails))
    # errors.append(evaluate(output, testCategory))
    # trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.100')
    # state = svm_train(trainMatrix, trainCategory)
    # output = svm_test(testMatrix, state)
    # errors.append(evaluate(output, testCategory))
    # trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.200')
    # state = svm_train(trainMatrix, trainCategory)
    # output = svm_test(testMatrix, state)
    # errors.append(evaluate(output, testCategory))
    # trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.400')
    # state = svm_train(trainMatrix, trainCategory)
    # output = svm_test(testMatrix, state)
    # errors.append(evaluate(output, testCategory))
    # trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.800')
    # state = svm_train(trainMatrix, trainCategory)
    # output = svm_test(testMatrix, state)
    # errors.append(evaluate(output, testCategory))
    # trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.1400')
    # state = svm_train(trainMatrix, trainCategory)
    # output = svm_test(testMatrix, state)
    # errors.append(evaluate(output, testCategory))
    #
    # x = [50, 100, 200, 400, 800, 1400]
    # y = errors
    #
    # plt.title('broadcast(b) vs join(r)')
    # plt.xlabel('Data Size')
    # plt.ylabel('Error(%)')
    # plt.plot(x, y)
    # plt.show()
    return


if __name__ == '__main__':
    main()
