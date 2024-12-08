from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#数据加载
data = pd.read_csv('data.txt', header=None)
data.columns = ['distance', 'weather', 'temperature', 'humidity', 'air_quality', 'default']
data['default'] = data['default'].map({'是': 1, '否': 0})

#特征和标签
X = data.drop('default', axis=1)
y = data['default']
X = pd.get_dummies(X)  #转换所有类别特征为数值（使用独热编码）

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
#预测
y_pred = rf_model.predict(X_test)

#评估模型
mse = mean_squared_error(y_test, y_pred)  # 均方误差
print(f"均方误差 (MSE): {mse}")

#设置画布尺寸
n_trees = len(rf_model.estimators_)  # 获取森林中的树的数量
n_cols = 5  # 设置每行显示多少棵树
n_rows = (n_trees // n_cols) + (n_trees % n_cols > 0)  # 计算需要的行数

# 创建子图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

# 将axes展平成一维数组，方便循环绘图
axes = axes.flatten()

# 绘制所有树
for i in range(n_trees):
    ax = axes[i]
    plot_tree(rf_model.estimators_[i], filled=True, feature_names=X.columns, class_names=['否', '是'], ax=ax, rounded=True)
    ax.set_title(f"Tree {i+1}")

# 隐藏多余的子图（如果有的话）
for i in range(n_trees, len(axes)):
    axes[i].axis('off')

# 显示图形
plt.tight_layout()
plt.show()

