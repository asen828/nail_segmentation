import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

#指甲分割模型
# class Nail_SegNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # ===== 编码器 =====
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2)

#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)

#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2)

#         self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
#         self.relu4 = nn.ReLU()

#         # ===== 解码器 =====
#         self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
#         self.relu5 = nn.ReLU()

#         self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
#         self.relu6 = nn.ReLU()

#         self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
#         self.conv7 = nn.Conv2d(32, 16, 3, padding=1)
#         self.relu7 = nn.ReLU()

#         # 输出层
#         self.out_conv = nn.Conv2d(16, 1, 1)

#     def forward(self, input):

#         # ===== 编码 =====
#         x1 = self.relu1(self.conv1(input))
#         p1 = self.pool1(x1)

#         x2 = self.relu2(self.conv2(p1))
#         p2 = self.pool2(x2)

#         x3 = self.relu3(self.conv3(p2))
#         p3 = self.pool3(x3)

#         x4 = self.relu4(self.conv4(p3))

#         # ===== 解码 =====
#         u1 = self.up1(x4)
#         if u1.shape != x3.shape:
#             u1 = F.interpolate(u1, size=x3.shape[2:])
#         u1 = torch.cat([u1, x3], dim=1)
#         u1 = self.relu5(self.conv5(u1))

#         u2 = self.up2(u1)
#         if u2.shape != x2.shape:
#             u2 = F.interpolate(u2, size=x2.shape[2:])
#         u2 = torch.cat([u2, x2], dim=1)
#         u2 = self.relu6(self.conv6(u2))

#         u3 = self.up3(u2)
#         if u3.shape != x1.shape:
#             u3 = F.interpolate(u3, size=x1.shape[2:])
#         u3 = torch.cat([u3, x1], dim=1)
#         u3 = self.relu7(self.conv7(u3))

#         out = self.out_conv(u3)

#         return out

def Nail_SegNet(encoder_name="resnet34", in_channels=3, classes=1):

    #定义Unet
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )
    
    weight_dir = ".vscode\segmentation\\resource\pretrain_weight"
    weight_name = "resnet34-333f7ec4.pth"

    local_weight_path = os.path.join(weight_dir, weight_name)
    # print(os.path.exists(local_weight_path))

    if os.path.exists(local_weight_path):
        print(f"[*] 找到本地预训练权重，正在加载: {local_weight_path}")

        state_dict = torch.load(
            local_weight_path, map_location="cpu", weights_only=False
        )

        model.encoder.load_state_dict(state_dict)
        print("[*] 预训练权重加载成功！")
    else:
        print(f"[!] 警告：未找到本地权重文件 {local_weight_path}")

    return model



def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)          # 将输出转为概率
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def calculate_metrics(preds, masks, threshold=0.5):
    """
    preds: 模型的原始输出 (logits)，未经过 sigmoid
    masks: 真实的二值化掩码 (0 或 1)
    """
    # 经过 Sigmoid 并二值化
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    masks = masks.float()

    # 1. Pixel Accuracy
    correct = (preds == masks).sum().item()
    total = torch.numel(preds)
    acc = correct / total

    # 2. IoU & Dice (平滑项 1e-6 防止除以 0)
    intersection = (preds * masks).sum().item()
    union = preds.sum().item() + masks.sum().item() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (preds.sum().item() + masks.sum().item() + 1e-6)

    return acc, iou, dice

def draw_data_curve(history):
    #根据历史数据绘制曲线图片
    epochs = range(1,len(history["avg_train_loss"]) + 1)
    
    plt.figure(figsize=(12,5))

    #左图 loss曲线
    plt.subplot(1,2,1) 
    line1 = plt.plot(epochs,history["avg_train_loss"],label="Train Loss")[0]
    line2 = plt.plot(epochs,history["avg_test_loss"],label="Test Loss")[0]
    plt.title("Training and Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend([line1,line2],["Training Loss","Test Loss"],loc = "lower left")

    #右图 评估曲线
    plt.subplot(1,2,2) 
    line1 = plt.plot(epochs,history["avg_acc"],label="Acc")[0]
    line2 = plt.plot(epochs,history["avg_iou"],label="Iou")[0]
    line3 = plt.plot(epochs,history["avg_dice"],label="Dice")[0]
    plt.title("Validation Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend([line1,line2,line3],["Acc","Iou","Dice"],loc = "lower right")

    plt.show()