import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt


def cal_entropy(data, x_labels):
    finish_columns = np.unique(data[:, -1])
    if len(finish_columns) == 1:
        return finish_columns[0]  # 返回ng与ok，表面这个分支已经分到叶子了
    dict_items = dict()  # 创建一个字典,储存有哪些类别
    len_num, len_label = data.shape  # 总类别数, 每一组总的个数
    len_label -= 1
    result = data[:, len_label]
    unique_decision = np.unique(result)  # ok, ng
    len_decision_ok = np.sum(result == unique_decision[0])  # ok的个数
    len_decision_nor = len(result) - len_decision_ok
    old_entropy = -float(len_decision_ok / len_num) * np.log2(float(len_decision_ok / len_num)) - float(len_decision_nor / len_num) * np.log2(float(len_decision_nor / len_num))

    message_entropy = []
    for i in range(len_label):
        item = data[:, i]
        unique_item = np.unique(item)  # 第i种类别的有哪些
        dict_items[x_labels[i]] = unique_item
        len_item = len(unique_item)  # 第i种类别的有多少种
        entropy_item = []
        entropy_item_len = []
        print("正在进行...")
        for j in range(len_item):
            # 遍历item，找到unique_item中每一个类别的下标
            indexs = np.where(item == unique_item[j])
            len_indexs = len(indexs[0])  # where返回的是一个元组，第一个元素是下标
            entropy_item_len.append(len_indexs)

            # 遍历result，找到对应的值
            result_item = result[indexs[0]]  # 使用 indexs[0]
            len_ok = np.sum(result_item == unique_decision[0])
            len_nor = len_indexs - len_ok

            # 计算信息熵
            if len_indexs > 0:  # 确保 len_indexs 不为零
                p1 = float(len_ok) / len_indexs
                p2 = float(len_nor) / len_indexs

                # 避免计算 log(0)
                entropy_value = 0
                if p1 > 0:
                    entropy_value -= p1 * np.log2(p1)
                if p2 > 0:
                    entropy_value -= p2 * np.log2(p2)

                entropy_item.append(entropy_value)
            else:
                entropy_item.append(0)

        # 计算加权熵
        entropy = 0
        for j in range(len_item):
            p = float(entropy_item_len[j]) / float(len_num)
            entropy += p * entropy_item[j]
        message_entropy.append(entropy)
    D = []
    for i in message_entropy:
        D.append(old_entropy - i)

    if not D:  # Check if D is empty
        return unique_decision[0]  # Return a default decision if no information gain

    max_index = np.argmax(D)  # 最大值的下标
    data_branch = dict_items[x_labels[max_index]]
    tree = {x_labels[max_index]: {}}
    for s in data_branch:
        next_indices = np.where(data[:, max_index] == s)[0]  # 生成式多维的
        next_data = data[next_indices, :]
        next_data = np.delete(next_data, max_index, axis=1)
        next_x_labels = np.delete(x_labels, max_index)
        tree[x_labels[max_index]][s] = cal_entropy(next_data, next_x_labels)
    return tree

# 绘制树状图
# 创建图
dot = Digraph()

# 定义决策树结构
node_counter = 0

def add_nodes_edges(tree, parent_name):
    global node_counter  # 引用全局变量

    for key, value in tree.items():

        if isinstance(value, dict):  # 如果是字典，说明还有子树
            node_counter += 1
            node_name = f"{parent_name}_{node_counter}"  # 生成唯一节点名称
            dot.node(node_name, key, fontname='SimHei')  # 添加节点
            dot.edge(parent_name, node_name)  # 连接边
            add_nodes_edges(value, node_name)  # 递归
        else:  # 否则是叶子节点
            node_counter += 1
            leaf_node_name = f"{parent_name}_{node_counter}"  # 生成唯一叶子节点名称
            dot.node(leaf_node_name, key, fontname='SimHei')  # 添加节点
            dot.edge(parent_name, leaf_node_name)  # 连接边

            node_counter += 1
            value_node_name = f"{leaf_node_name}_{node_counter}"  # 生成唯一叶子节点名称
            dot.node(value_node_name, value, fontname='SimHei')  # 添加叶子节点
            dot.edge(leaf_node_name, value_node_name)  # 连接边


def predict_single(tree, x_labels, sample):
    if not isinstance(tree, dict):
        return tree == sample[-1]  # 比较预测结果和实际结果是否相同
    root = next(iter(tree))
    subtree = tree[root]
    feature_index = list(x_labels).index(root)  # Convert x_labels to list before using index
    feature_value = sample[feature_index]
    if feature_value in subtree:
        return predict_single(subtree[feature_value], x_labels, sample)
    else:
        # 返回叶子节点的预测结果，并比较该结果与实际结果是否相同
        leaf_prediction = next(iter(subtree.values()))
        return leaf_prediction == sample[-1]

def predict(tree, x_labels, samples):
    correct_predictions = 0
    for sample in samples:
        if predict_single(tree, x_labels, sample):
            correct_predictions += 1
    return correct_predictions / len(samples)  # 返回预测正确的比例




if __name__ == "__main__":

    with open('data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()[:8]
    data = np.array([i.strip().split(',') for i in data])
    x_labels=np.array(["距离", "天气", "安全性","费用", "环境质量"])
    # 矩阵进行转置,得到data
    # 调用cal_entropy函数
    tree = cal_entropy(data, x_labels)
    # 可视化决策树
    # 画决策树
    dot.attr(fontname='SimHei')
    # 画决策树
    add_nodes_edges(tree, 'root')  # 假设 'root' 是根节点的名称
    # 渲染图像
    dot.render('decision_tree', format='png', view=True)
    #预测
    with open('data_test.txt', 'r', encoding='utf-8') as f:
        test_data = f.readlines()
        test_data = np.array([i.strip().split(',') for i in test_data])
    accuracy = predict(tree, x_labels, test_data)
    print(f"预测准确率: {accuracy}")






