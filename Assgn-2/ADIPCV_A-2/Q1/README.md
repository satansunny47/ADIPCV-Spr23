# Train CIFAR10 with PyTorch

This the Image Classification problem using CIFAR10 dataset. The train and test datasets contain 48000 and 12000 images each. Each dataset has 10 classes with equal number of images.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# The model to be used can be seen in line 154, 
net = models.resnet50(pretrained=True) 
net = models.efficientnet_b0(pretrained=True)

Just comment out the model which is not to be used.

# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |

| [ResNet-50]               | 63.812%      |
| [EfficientNet-b0]         | 60.235%      |


All models are trained for 50 epochs.

The outputs of the above in terminal are given as text in the outputs file.

