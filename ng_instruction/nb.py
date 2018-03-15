import numpy as np
from ng_instruction import svm


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    print(file, ': ', rows, cols)
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for row_index, line_data in enumerate(fd):
        nums = [int(x) for x in line_data.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        # 累计求和
        k = np.cumsum(kv[:-1:2])
        # 从第二个元素开始，每隔两个取一个
        v = kv[1::2]
        matrix[row_index, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    feature_count = matrix.shape[1]
    ###################
    state = svm.svm_train(matrix, category)
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    output = svm.svm_test(matrix, state)
    ###################
    return output


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print ('Error: %1.4f' % error)

def main():
    trainMatrix, tokenlist, trainCategory = svm.readMatrix('MATRIX.TRAIN.400')
    testMatrix, tokenlist, testCategory = svm.readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
