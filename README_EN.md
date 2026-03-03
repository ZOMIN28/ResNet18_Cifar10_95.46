# ResNet18_CIFAR10_95.46

[中文版本文档](./README.md)

**PyTorch implementation:** Training a ResNet18 model on the CIFAR-10 dataset from scratch (no pretrained weights), achieving **95.46%** test accuracy.

**Author:** ZOMIN
 GitHub: https://github.com/ZOMIN28



This project demonstrates how to train a modified ResNet18 on CIFAR-10 using data augmentation and architectural adjustments. Without any pretrained parameters, the model achieves **95.46%** accuracy on the test set. The same approach also reaches **95.46%** accuracy on the 300,000-image CIFAR-10 dataset used in the Kaggle competition.

------

## 1. CIFAR-10 Dataset

CIFAR-10 consists of **60,000** 32×32 RGB images across **10 classes**, with:

- 50,000 training images
- 10,000 test images

In PyTorch, the dataset can be directly loaded using:

```
torchvision.datasets.CIFAR10()
```

------

## 2. Data Augmentation

To improve generalization and reduce overfitting, the following data augmentation techniques are applied during training:

### (1) Random Horizontal Flip

```
transforms.RandomHorizontalFlip()
```

### (2) Random Crop with Padding

Images are padded from 32×32 to 40×40, then randomly cropped back to 32×32:

```
transforms.RandomCrop(32, padding=4)
```

### (3) Cutout

Cutout randomly masks square regions in the image. In this project, we use:

- Number of holes: 1
- Length: 16 pixels

```
Cutout(n_holes=1, length=16)
```

Official Cutout implementation:
 https://github.com/uoguelph-mlrg/Cutout

------

## 3. ResNet18 Modifications

Since CIFAR-10 images are small (32×32), the original 7×7 convolution and max pooling in ResNet18 may discard too much spatial information. Therefore:

- Replace the 7×7 convolution with a 3×3 convolution
- Remove the max pooling layer
- Use stride=1 and padding=1 to preserve spatial details

Modified first convolution layer:

```
model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
```

Max pooling layer removed in the forward pass:

```
def _forward_impl(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    # x = self.maxpool(x)

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

------

## 4. Training Strategy

- Initial learning rate: **0.1**
- If validation loss does not decrease for 10 epochs, reduce learning rate by **0.5**
- Total training epochs: **250**
- Batch size: **128**
- Optimizer: **SGD** with momentum and weight decay

```
optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4
)
```

------

## 5. Training and Testing Code

The complete code is available on GitHub.

**Note:** Before running the code, create a folder named `checkpoint` in the project directory to store model weights.

### Pretrained Model Download

Baidu Drive:
 https://pan.baidu.com/s/1yKXWWf1UEXS_gsWnM6sFDA
 Extraction code: `z66g`

Google Drive:
 https://drive.google.com/file/d/1AEYfIHDaNIZywkYKsiaNVpfr2hAJ1Fti/view

------

## 6. Environment

```
python == 3.6
torch == 1.10.2
torchvision == 0.11.3
numpy == 1.19.5
```