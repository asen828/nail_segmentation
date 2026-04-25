import numpy as np
import torch
import os
from dataset import *
from model import *
from torch.utils.tensorboard import SummaryWriter


root_dir = "resource"  #数据集根路径
image_size = (256,256)      #图片缩放256*256
train_rate = 0.8            #取数据集的80%，做训练
batch_size = 8              #批大小
lr         = 1e-4           #学习率

train_loader,test_loader = get_dataloader(root_dir,batch_size,image_size,train_rate)

model = Nail_SegNet()   #分割模型
criterion = nn.BCEWithLogitsLoss() #损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #训练设备
model = model.to(device)
criterion = criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)   #优化器
train_epoch = 600  #训练轮次
total_train_step = 0
total_test_step = 0

history = {
    "avg_train_loss":[],
    "avg_test_loss":[],
    "avg_acc":[],
    "avg_iou":[],
    "avg_dice":[]
}

best_val_iou = 0.0

for epoch in range(train_epoch):
    print(f"--------epoch:{epoch}-----------")
    #训练模型
    train_loss = 0.0
    model.train()
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_train_step += 1
        # if(total_train_step % 100 == 0):
        #  print(f"-----第{total_train_step}训练，loss = {loss.item():.4f}------")
    avg_train_loss = train_loss/len(train_loader)
    #模型测试
    model.eval()
    total_test_loss = 0.0
    val_acc, val_iou, val_dice = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            # print(output)
            loss = criterion(output,labels)
            total_test_loss += loss.item()
            
            acc, iou, dice = calculate_metrics(output, labels)
            val_acc += acc
            val_iou += iou
            val_dice += dice
    avg_test_loss = total_test_loss / len(test_loader)   #验证集平均损失
    avg_acc = val_acc / len(test_loader)                 #验证集的平均准确率
    avg_iou = val_iou / len(test_loader)                 #验证集的平均交并比
    avg_dice = val_dice / len(test_loader)               #验证集的dice系数
    
    #打印测试数据
    print(f"avg_train_loss={avg_train_loss:.6f}")
    print(f"avg_test_loss={avg_test_loss:.6f}")
    print(f"avg_acc={avg_acc:.6f}")
    print(f"avg_iou={avg_iou:.6f}")
    print(f"avg_dice={avg_dice:.6f}")

    #存储测试数据
    history["avg_train_loss"].append(avg_train_loss)
    history["avg_test_loss"].append(avg_test_loss)
    history["avg_acc"].append(avg_acc)
    history["avg_iou"].append(avg_iou)
    history["avg_dice"].append(avg_dice)

    #保存优秀的模型
    if(avg_iou > 0.7 and avg_iou > best_val_iou):
        best_val_iou = avg_iou
        torch.save(model,f"resource\models\segnet_{avg_iou}.pth")
draw_data_curve(history)        #保存曲线



