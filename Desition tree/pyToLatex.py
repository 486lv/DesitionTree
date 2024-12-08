import numpy as np
import numpy as np

data = np.array([
    ["远", "雨天", "高", "少", "一般", "否"],
    ["近", "阴天", "低", "少", "良好", "否"],
    ["适中", "大风", "低", "适中", "良好", "是"],
    ["远", "晴天", "中", "少", "良好", "否"],
    ["近", "晴天", "高", "多", "良好", "是"],
    ["适中", "阴天", "低", "少", "良好", "否"],
    ["远", "大风", "低", "多", "一般", "否"],
    ["近", "雨天", "低", "适中", "良好", "是"],
    ["适中", "晴天", "中", "适中", "良好", "否"],
    ["远", "大风", "高", "适中", "一般", "否"],
    ["远", "晴天", "高", "少", "良好", "是"],
    ["近", "阴天", "低", "少", "良好", "是"]
])

print(data)

x_labels = ["距离", "天气", "安全性","费用", "环境质量"]



# 将数据转换为 LaTeX 表格
def generate_latex_table(data, x_labels):
    # 生成表头
    header = " & ".join(x_labels) + " \\\\ \\hline\n"

    # 生成表格的每一行
    rows = []
    for row in data:
        row_str = " & ".join(row) + " \\\\"
        rows.append(row_str)

    # 合并表头和数据行
    table = "\\begin{tabular}{|" + "c|" * len(x_labels) + "}\n\\hline\n" + header + "\n".join(rows) + "\n\\hline\n\\end{tabular}"

    return table

# 生成并打印 LaTeX 表格
latex_table = generate_latex_table(data, x_labels)
print(latex_table)

