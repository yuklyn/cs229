import numpy as np

TAG = 'yuklyn_SVM'
DEBUG = True

Linear = 1
RBF = 2


def log(instruction='', *content):
    if DEBUG:
        print(TAG, instruction, end=' ')
        for item in content:
            print(item, end=' ')
        print('')


class SVM:
    def __init__(self, tolerance=0.001, epsilon=0.001, penalty_parameter=100.0, kernel=Linear):
        # 违反KKT条件的范围
        self.tolerance = tolerance
        # alpha更新大小
        self.epsilon = epsilon
        # 惩罚系数
        self.penalty_parameter = penalty_parameter * 1.0

        self.kernel = kernel
        self.kernel_values = np.array(0)

        self.alpha = np.array(0)
        self.errors = np.array(0)
        self.b = 0.0
        self.train_set_x = np.array(0)
        self.train_set_y = np.array(0)

        self.support_set_x = np.array(0)
        self.support_set_y = np.array(0)
        self.support_alpha = np.array(0)

        self.loop_count = 5
        self.delta = 8

        self.ignore_iterate_count = 0

    def compute_kernel_values(self, data_set_x):
        if self.kernel == Linear:
            self.kernel_values = data_set_x.dot(data_set_x.T)
        elif self.kernel == RBF:
            squared = np.sum(data_set_x * data_set_x, axis=1)
            gram = data_set_x.dot(data_set_x.T)
            self.kernel_values = np.exp(
                -(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (self.delta ** 2)))
        else:
            log('未知核函数')

    def get_kernel_value(self, i, j=None):
        if j is None and i is None:
            return self.kernel_values
        elif i is None:
            return self.kernel_values[:, j]
        elif j is None:
            return self.kernel_values[i, :]
        else:
            return self.kernel_values[i][j]

    def kernel_func(self, x_i, x_j):
        if self.kernel == Linear:
            return x_i.dot(x_j.T)
        elif self.kernel == RBF:
            return np.exp(-((x_i - x_j).dot((x_i - x_j).T)) / (2 * (self.delta ** 2)))
        else:
            log('未知核函数')

    def learn_func(self, j):
        # 直接从118下降到20-40
        return (self.alpha * self.train_set_y).dot(self.get_kernel_value(j).T) + self.b

    def compute_error(self, y_j, j):
        self.errors[j] = self.learn_func(j) - y_j

    def outer_loop(self, i):
        y_i = self.train_set_y[i]
        alpha_i = self.alpha[i]
        # !!!!!!!原来问题在这，误差要重新算一遍
        self.compute_error(y_i, i)
        error_i = self.errors[i]
        r_i = y_i * error_i

        if (r_i > self.tolerance and alpha_i > 0) or (r_i < -self.tolerance and alpha_i < self.penalty_parameter):
            return self.inner_loop(i)

        return 0

    def inner_loop(self, i):
        error_i = self.errors[i]
        j = -1

        # 寻找|Ei - Ej|最大的j
        minor = 0
        for index, error in enumerate(self.errors):
            if index == i:
                continue
            minor_temp = abs(error_i - error)
            if minor_temp > minor:
                minor = minor_temp
                j = index
        if j >= 0:
            if self.iterate(i, j):
                return 1

        # 在支持向量上寻找alpha_j
        for index, alpha in enumerate(self.alpha):
            if 0 < alpha < self.penalty_parameter and index != i:
                if self.iterate(i, index):
                    return 1

        # 在所有向量上寻找alpha_j
        random_index = int(np.random.rand() * len(self.train_set_y))
        for index, alpha in enumerate(self.alpha):
            real_index = (index + random_index) % len(self.train_set_y)
            if real_index == i or 0 < alpha < self.penalty_parameter:
                continue
            if self.iterate(i, real_index):
                return 1

        return 0

    def iterate(self, i, j):
        y_i = self.train_set_y[i]
        y_j = self.train_set_y[j]
        alpha_i = self.alpha[i]
        alpha_j = self.alpha[j]
        error_i = self.errors[i]
        error_j = self.errors[j]

        if y_i != y_j:
            low_margin = max(0.0, alpha_j - alpha_i)
            high_margin = min(self.penalty_parameter, self.penalty_parameter + alpha_j - alpha_i)
        else:
            low_margin = max(0.0, alpha_j + alpha_i - self.penalty_parameter)
            high_margin = min(self.penalty_parameter, alpha_j + alpha_i)

        if low_margin == high_margin:
            return 0

        kii = self.get_kernel_value(i, i)
        kjj = self.get_kernel_value(j, j)
        kij = self.get_kernel_value(i, j)
        eta = kii + kjj - 2 * kij

        if eta > 0:
            alpha_j_new = alpha_j + (y_j * (error_i - error_j) * 1.0 / eta)
            if alpha_j_new > high_margin:
                alpha_j_new = high_margin
            elif alpha_j_new < low_margin:
                alpha_j_new = low_margin
        else:
            return 0

        if abs(alpha_j_new - alpha_j) < self.epsilon:
            if alpha_j_new != 0:
                self.ignore_iterate_count += 1
            return 0

        alpha_i_new = alpha_i + y_i * y_j * (alpha_j - alpha_j_new)

        # 更新b
        b_i = -error_i - y_i * kii * (alpha_i_new - alpha_i) - y_j * kij * (alpha_j_new - alpha_j) + self.b
        b_j = -error_j - y_i * kij * (alpha_i_new - alpha_i) - y_j * kjj * (alpha_j_new - alpha_j) + self.b
        if 0 < alpha_i_new < self.penalty_parameter:
            self.b = b_i
        elif 0 < alpha_j_new < self.penalty_parameter:
            self.b = b_j
        else:
            self.b = (b_i + b_j) * 0.5

        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        self.compute_error(y_i, i)
        self.compute_error(y_j, j)

        return 1

    def smo_algorithm(self):
        iterate_all = True
        changed_alpha_count = 0
        iterate_all_count = 0
        loop_count = 0
        while changed_alpha_count > 0 or iterate_all:
            loop_count += 1
            log('loop_count: ', loop_count, ' iterate_all_count: ', iterate_all_count)

            changed_alpha_count = 0
            if iterate_all:
                for index, alpha_i in enumerate(self.alpha):
                    changed_alpha_count += self.outer_loop(index)
            else:
                for index, alpha_i in enumerate(self.alpha):
                    if 0 < alpha_i < self.penalty_parameter:
                        changed_alpha_count += self.outer_loop(index)
            if iterate_all is True:
                iterate_all = False
            if changed_alpha_count == 0:
                iterate_all_count += 1
                iterate_all = True
            if iterate_all_count >= self.loop_count:
                break

        indexes = np.where(self.alpha != 0)
        self.support_alpha = self.alpha[indexes]
        self.support_set_x = self.train_set_x[indexes]
        self.support_set_y = self.train_set_y[indexes]

        log('支持向量的个数: ', len(self.support_alpha))
        log('支持向量: \n', self.support_alpha)
        log('b: ', self.b)

        log('忽略的迭代次数: ', self.ignore_iterate_count)

    def train(self, data_set_x, data_set_y):
        sample_count, feature_count = np.shape(data_set_x)
        self.alpha = np.zeros(sample_count)
        self.errors = np.zeros(sample_count)

        self.train_set_x = data_set_x
        self.train_set_y = data_set_y

        self.kernel_values = np.zeros((sample_count, sample_count))
        self.compute_kernel_values(data_set_x)

        for i in range(len(self.alpha)):
            self.compute_error(self.train_set_y[i], i)

        self.smo_algorithm()

    def test(self, data_set_x, data_set_y):
        output = self.predict(data_set_x)
        fails = np.where(np.sign(output) != data_set_y)
        log('分类错误的样本个数: ', len(fails[1]))
        log('分类错误的样本索引: \n', fails[1])
        return len(fails[0])

    def test_kernel_func(self, test_set_x):
        if self.kernel == Linear:
            return test_set_x.dot(self.support_set_x.T)
        elif self.kernel == RBF:
            if len(np.shape(test_set_x)) < 2:
                test_set_x = test_set_x.reshape(1, -1)
            squared_i = np.sum(test_set_x * test_set_x, axis=1)
            squared_j = np.sum(self.support_set_x * self.support_set_x, axis=1)
            gram = test_set_x.dot(self.support_set_x.T)
            return np.exp(
                -(squared_i.reshape((-1, 1)) + squared_j.reshape((1, -1)) - 2 * gram) / (2 * (self.delta ** 2)))
        else:
            log('未知核函数')

    def predict(self, data_set_x):
        return (self.support_alpha * self.support_set_y.reshape(1, -1))\
                   .dot(self.test_kernel_func(data_set_x).T) + self.b

    def test_error(self, x_j, y_j):
        return self.predict(x_j) - y_j

    def compute_omega(self):
        omega = 0.0
        for i, alpha in enumerate(self.support_alpha):
            omega += alpha * self.support_set_y[i] * self.support_set_x[i]
        return omega
