import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from model import * 

image_path = "resource\\nail_3.jpg"
model_path = "resource\models\segnet_0.7719546698783288.pth"

image = cv2.imread(image_path)
image = cv2.resize(image,(256,256))

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
trans_image = trans(image)

#加载模型
deive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nail_model = torch.load(model_path,weights_only=False)
trans_image = torch.reshape(trans_image,(1,-1,256,256))
Nail_model = Nail_model.to(deive)
trans_image = trans_image.to(deive)

Nail_model.eval()
with torch.no_grad():
    output = Nail_model(trans_image)
# print(output)

#图像可视化
prob = torch.sigmoid(output)
mask = (prob>0.5).float().cpu().squeeze().numpy()
mask_uint8 = (mask*255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
mask_uint8 = cv2.erode(mask_uint8,kernel,iterations=3)      #去除小杂点
mask_uint8 = cv2.dilate(mask_uint8,kernel,iterations=3)

counters,hierarchy = cv2.findContours(mask_uint8,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #绘制边框
cv2.drawContours(image,counters,-1,(255,0,0),2)



cv2.imshow("mask",mask_uint8)
cv2.imshow("image",image)
cv2.waitKey(0)
