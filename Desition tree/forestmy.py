import numpy as np

class SimpleDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # 如果数据集太小，或者到达最大深度，创建叶子节点
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth == self.max_depth):
            return np.mean(y)

        # 找到最佳分裂点
        best_split = self._best_split(X, y)

        # 如果没有找到有效的分裂点，则返回叶子节点
        if best_split is None:
            return np.mean(y)

        left_mask = X[:, best_split['feature']] <= best_split['threshold']
        right_mask = ~left_mask

        # 递归构建左右子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_split['feature'], 'threshold': best_split['threshold'],
                'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_split = None
        best_score = float('inf')

        # 遍历所有特征，寻找最佳分裂点
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # 如果某个子集的样本数小于2，跳过
                if len(y[left_mask]) < 2 or len(y[right_mask]) < 2:
                    continue

                left_score = np.mean((y[left_mask] - np.mean(y[left_mask]))**2)
                right_score = np.mean((y[right_mask] - np.mean(y[right_mask]))**2)
                score = left_score + right_score

                if score < best_score:
                    best_score = score
                    best_split = {'feature': feature, 'threshold': threshold}

        return best_split

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if isinstance(tree, dict):
            if sample[tree['feature']] <= tree['threshold']:
                return self._predict_sample(sample, tree['left'])
            else:
                return self._predict_sample(sample, tree['right'])
        else:
            return tree

# 测试
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 2, 3, 4])
regressor = SimpleDecisionTreeRegressor(max_depth=2)
regressor.fit(X, y)
print(regressor.predict(X))
