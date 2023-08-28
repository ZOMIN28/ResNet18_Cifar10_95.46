# ResNet18_Cifar10_95.46
Pytorch实现：使用ResNet18网络训练Cifar10数据集，测试集准确率达到95.46%(从0开始，不使用预训练模型)
### `Pytorch`实现：使用`ResNet18`网络训练`Cifar10`数据集，测试集准确率达到95.46%(从0开始，不使用预训练模型)

作者：`ZOMIN`：[ZOMIN28 (github.com)](https://github.com/ZOMIN28)

本文将介绍如何使用数据增强和模型修改的方式，在不使用任何预训练模型参数的情况下，在`ResNet18`网络上对`Cifar10`数据集进行分类任务。在测试集上，我们的模型准确率可以达到95.46%。在`Kaggle`的`Cifar10`比赛上，我训练的模型在300,000的超大`Cifar10`数据集上依然可以达到95.46%的准确率。

#### 1 `Cifar10`数据集

`Cifar10`数据集由10个类的60000个尺寸为`32x32`的`RGB`彩色图像组成，每个类有6000个图像， 有50000个训练图像和10000个测试图像。

在使用`Pytorch`时，我们可以直接使用`torchvision.datasets.CIFAR10()`方法获取该数据集。

#### 2 数据增强

为了提高模型的泛化性，防止训练时在训练集上过拟合，往往在训练的过程中会对训练集进行数据增强操作，例如随机翻转、遮挡、填充后裁剪等操作。我们这里对训练集做如下三种处理：

##### (1)随机翻转
代码如下：

```python
transforms.RandomHorizontalFlip()
```

##### (2)填充后随机裁剪

我们可以将尺寸为`32x32`的图像填充为`40x40`，然后随机裁剪成`32x32`。
```python
transforms.RandomCrop(32, padding=4)
```

##### (3)Cutout操作

Cutout操作会随机遮挡图片的若干尺寸的若干块，尺寸和块可以根据自己的需要设置。

调用代码如下，这里我们设置块为1，尺寸长度为16个像素。cutout的完整操作将在后面给出。Github链接：https://github.com/uoguelph-mlrg/Cutout

```python
Cutout(n_holes=1, length=16)
```

#### 3 修改`ResNet18`模型

考虑到`CIFAR10`数据集的图片尺寸太小，`ResNet18`网络的`7x7`降采样卷积和池化操作容易丢失一部分信息，所以在实验中我们将`7x7`的降采样层和最大池化层去掉，替换为一个`3x3`的降采样卷积，同时减小该卷积层的步长和填充大小，这样可以尽可能保留原始图像的信息。

修改卷积层如下：

```python
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
```

删去最大池化层：

```python
def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
def forward(self, x):
        return self._forward_impl(x)
```

#### 4 训练策略

在模型的训练上，我们采用的策略是：设置初始学习率为0.1，每当经过10个epoch训练的验证集损失没有下降时，学习率变为原来的0.5，共训练250个epoch。在训练中，我们的batch_size大小为128，优化器为`SGD`：

```python
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
```

#### 5 训练和测试代码

完整的代码已上传至我的`github`
注意：若要运行代码，需要在项目文件夹中创建名为checkpoint的文件夹，用于存放参数文件。
预训练模型链接<br/>
百度云盘：<br/>
https://pan.baidu.com/s/1yKXWWf1UEXS_gsWnM6sFDA  提取码：z66g。<br/>
Google云盘：<br/>
https://drive.google.com/file/d/1AEYfIHDaNIZywkYKsiaNVpfr2hAJ1Fti/view?usp=drive_link

#### 6 版本
python == 3.6 <br/>
torch == 1.10.2 <br/>
torchvision == 0.11.3 <br/>
numpy == 1.19.5 <br/>
