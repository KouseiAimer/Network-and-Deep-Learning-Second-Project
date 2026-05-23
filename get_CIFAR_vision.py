import torchvision
import matplotlib.pyplot as plt

# =========================
# 1. 加载 CIFAR-10 数据集
# =========================
data_dir = "./data"

dataset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,        # 可改成 False，看测试集
    download=False,    # 你已经下载好了，所以这里用 False
    transform=None     # 保持 PIL Image，方便可视化
)

classes = dataset.classes

print(classes)
print("数据集大小:", len(dataset))


# =========================
# 2. 每个类别选若干张图片
# =========================
num_images_per_class = 10

selected_images = {i: [] for i in range(10)}

for img, label in dataset:
    if len(selected_images[label]) < num_images_per_class:
        selected_images[label].append(img)

    # 如果每个类别都已经选够了，就停止
    if all(len(selected_images[i]) == num_images_per_class for i in range(10)):
        break


# =========================
# 3. 画出类似 CIFAR-10 Dataset 的图
# =========================
fig = plt.figure(figsize=(12, 8))

# 10 行，11 列
# 第 1 列放类别名称，后 10 列放图片
gs = fig.add_gridspec(
    nrows=10,
    ncols=11,
    width_ratios=[2.2] + [1] * 10,
    wspace=0.05,
    hspace=0.05
)

for row in range(10):
    # 左侧类别名称
    ax_label = fig.add_subplot(gs[row, 0])
    ax_label.text(
        1.0, 0.5,
        classes[row],
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="center"
    )
    ax_label.axis("off")

    # 当前类别的图片
    for col in range(num_images_per_class):
        ax_img = fig.add_subplot(gs[row, col + 1])
        ax_img.imshow(selected_images[row][col])
        ax_img.axis("off")

# 添加标题
fig.text(
    0.5, 0.03,
    "Figure 1: CIFAR-10 Dataset",
    ha="center",
    fontsize=20
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()