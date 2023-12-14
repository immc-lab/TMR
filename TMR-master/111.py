import numpy as np

# 创建训练集和测试集
X_train = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
y_train = np.array([1, 2, 3, 4])
X_test = np.array([[1, 1], [4, 4]])
k = 3


# 创建KNN模型
class KNN:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        y_pred = np.zeros(X.shape[0], dtype=self.y_train.dtype)
        for i, x_test in enumerate(X):
            # 计算距离
            dists = np.sqrt(np.sum((x_test - self.X_train) ** 2, axis=1))
            # 找到最近的前k个样本
            sorted_index = np.argsort(dists)
            top_K = sorted_index[:k]
            closest_y = self.y_train[top_K]
            # 投票统计
            y_pred[i] = np.argmax(np.bincount(closest_y)) #存在一样多的情况
        return y_pred


# 创建KNN实例并进行预测
knn = KNN()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test, k)

print(y_pred)
