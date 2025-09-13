# import numpy as np
# import matplotlib.pyplot as plt

# # 加载 .npz 文件
# data = np.load('train.npz')
# images = data['images']
# labels = data['labels']

# print(f"Images shape: {images.shape}")
# print(f"Labels shape: {labels.shape}")

# # 遍历并显示每张图像
# for i in range(len(images)):
#     img = np.moveaxis(images[i], 0, -1)  # 把通道轴移到最后
#     plt.imshow(img)
#     plt.title(f"Label: {labels[i]}")
#     plt.axis('off')
#     plt.show()  # 显示图像并等待窗口关闭后继续

import numpy as np
import matplotlib.pyplot as plt

# 加载 .npz 文件
data = np.load('train.npz')
images = data['images']
labels = data['labels']

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# 找出所有唯一的 label
unique_labels = np.unique(labels)

for label in unique_labels:
    # 找出当前 label 对应的所有索引
    idxs = np.where(labels == label)[0]
    n = len(idxs)

    # 打印该 label 有多少张图
    print(f"Label {label}: {n} image(s)")

    # 设置子图行列数
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    # 创建一个新图像窗口
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(f"Label: {label}", fontsize=16)

    # 展示每张图像
    for i, idx in enumerate(idxs):
        img = np.moveaxis(images[idx], 0, -1)
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i % cols]
        ax.imshow(img)
        ax.axis('off')

    # 关闭多余子图（若有）
    for j in range(n, rows * cols):
        ax = axs[j // cols, j % cols] if rows > 1 else axs[j % cols]
        ax.axis('off')

    plt.tight_layout()
    plt.show()
