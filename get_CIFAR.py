import torchvision
import torchvision.transforms as transforms

# 数据集保存路径
data_dir = "./data"

# 定义基础 transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# 下载 CIFAR-10 训练集
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=transform
)

# 下载 CIFAR-10 测试集
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=transform
)

print("CIFAR-10 数据集下载完成！")
print("训练集大小:", len(train_dataset))
print("测试集大小:", len(test_dataset))
print("类别:", train_dataset.classes)
