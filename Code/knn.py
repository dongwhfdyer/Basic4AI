"""
KNN
分别实现线性搜索以及kd树搜索
"""

import heapq

import numpy as np
from sklearn.datasets import load_digits
from tqdm import tqdm


class Node(object):
    '''
    树节点
    One Node ,one data.
    '''

    def __init__(self, data, left, right, dim):
        self.data = data  # 节点数据
        self.left = left  # 左节点
        self.right = right  # 右节点
        self.dim = dim  # 节点对应数据维度


class KDTree(object):
    '''
    kd树
    '''

    def __init__(self, datas):
        '''
        datas 特征以及标签，最后一列是标签值，方便最后进行分类
        '''
        self.root = self.build_tree(datas, 0, datas.shape[1] - 1)  # 根节点

    def build_tree(self, datas, cur_dim, max_dim):  # max_dim means the dimension of the data feature
        '''
        构建kd树
        '''
        if len(datas) == 0:
            return
        new_datas = datas[np.argsort(datas[:, cur_dim])]  # current dimension. The split point is the median of the current dimension. So they used argsort to sort the data.
        mid = datas.shape[0] // 2
        return Node(datas[mid], self.build_tree(datas[:mid], (cur_dim + 1) % max_dim, max_dim),  # recursively build left and right subtree
                    self.build_tree(datas[mid + 1:], (cur_dim + 1) % max_dim, max_dim), cur_dim)

    def predict(self, x, k, lp_distance):
        '''
        使用kd树进行预测
        k means class number
        x means the test data waiting to be classified
        '''
        top_k = [(-np.inf, None)] * k  # original k is 5

        # 递归访问节点
        def visit(node):
            if node is None:
                return
            ################################################## This two line will let the whole process start from the leaf node which is the nearst one.
            dis_with_axis = x[node.dim] - node.data[node.dim]  # node.data[node.dim] is the median of the current dimension or it's called the first element of the current dimension.
            visit(node.left if dis_with_axis < 0 else node.right)  # visit left subtree if x[node.dim] < node.data[node.dim]. It means that if the object data of current dimension is less than the current dimension's splitting point or median, then the object data is in the left subtree.
            ##################################################

            dis_with_node = lp_distance(x.reshape((1, -1)), node.data.reshape((1, -1))[:, :-1])[0]  # calculate the distance between x and node.data. x.shape = (1, n), node.data.shape = (1, n+1). So the distance is calculated between x and node.data.
            # For `heapq.heappushpop` if (***) is less than top_k heap, then do nothing. If (***) is larger than  the first element of top_k heap, then pop the first element and push the (***) into t  # if (***) is less than top_k heap, then do nothing. If (***) is larger than  the first element of top_k heap, then pop the first element and push the (***) into top_k heap.op_k heap.
            # first example: (-inf, None)<(54,7)
            heapq.heappushpop(top_k, (-dis_with_node, node.data[-1]))  # when compare the tuple, the first element has the higher priority. So, this line means push the the element with the smaller(it's done by "-") distance into top_k heap.

            # why there's "-", heapq.heappushpop() is a min heap. So, the first element is the smallest element. However and the above function heappushpop will only push when the element is larger than the first element of top_k heap.
            if -top_k[0][0] > abs(dis_with_axis):  # if the distance is larger than the distance with axis, then visit right subtree.
                visit(node.right if dis_with_axis < 0 else node.left)

        visit(self.root)

        top_k = [int(x[1]) for x in heapq.nlargest(k, top_k)]

        return top_k


class KNN(object):

    def __init__(self, k, train_xs, train_ys, p=2):
        '''
        k KNN的k值
        train_xs np.array 训练集样本特征
        train_ys np.array 训练集样本标签
        p Lp 距离的p值
        '''
        self.k = k

        self.train_xs = train_xs
        self.train_ys = train_ys
        self.p = p

        self.kdtree = None

    def lp_distance(self, x1, x2):
        '''
        Lp距离
        x1.shape can be (num_sample, feature_num)
        x2.shape can be (1, feature_num)
        p stands for the order of Lp
        so it will calculate the distance between (each element in x1) and (x2)
        '''
        dis = np.sum(np.abs(x1 - x2) ** self.p, -1) ** (1 / self.p)  # |x1-x2|^p
        return dis

    def liner_test(self, test_xs):
        '''
        线性搜索测试
        '''
        predict_ys = []
        print('Testing.')
        for test_x in tqdm(test_xs):
            dis = self.lp_distance(self.train_xs, test_x.reshape((1, -1)))  # test_x.shape = (1, feature_num). test_x is only one sample. train_xs.shape = (num_sample, feature_num)
            top_k_index = dis.argsort()[:self.k]  # it will sort the distance and return the index of the top k nearest samples.
            top_k = [self.train_ys[index] for index in top_k_index]  # it will return the label of the top k nearest samples.
            predict_y = self.vote(top_k)
            predict_ys.append(predict_y)
        predict_ys = np.array(predict_ys)
        return predict_ys

    def kdtree_test(self, test_xs):
        '''
        kd树搜索测试
        '''
        if self.kdtree is None:
            self.kdtree = KDTree(np.concatenate([self.train_xs, self.train_ys.reshape((-1, 1))], -1))  # shape = [1437,65]

        predict_ys = []
        for test_x in tqdm(test_xs):
            top_k = self.kdtree.predict(test_x, self.k, self.lp_distance)
            predict_y = self.vote(top_k)
            predict_ys.append(predict_y)
        predict_ys = np.array(predict_ys)

        return predict_ys

    def vote(self, top_k):
        '''
        多数表决
        '''
        count = {}
        max_freq, predict_y = 0, 0
        for key in top_k:  # if in the original example, we assume top_k has 5 elements. Then they will vote for the 5 classes.

            count[key] = count.get(key, 0) + 1

            if count[key] > max_freq:
                max_freq = count[key]
                predict_y = key
        return predict_y


if __name__ == '__main__':
    # 加载sklearn自带的手写数字识别数据集
    digits = load_digits()
    features = digits.data
    targets = digits.target

    # 随机打乱数据
    shuffle_indices = np.random.permutation(features.shape[0])
    features = features[shuffle_indices]
    targets = targets[shuffle_indices]

    # 划分训练、测试集
    train_count = int(len(features) * 0.8)
    train_xs, train_ys = features[:train_count], targets[:train_count]
    test_xs, test_ys = features[train_count:], targets[train_count:]

    k = 5
    p = 2
    knn = KNN(k, train_xs, train_ys, p)

    # predict_ys = knn.liner_test(test_xs)
    predict_ys = knn.kdtree_test(test_xs)

    accuracy = (predict_ys == test_ys).sum() / test_ys.shape[0]

    print('Accuracy:%.4f' % accuracy)
