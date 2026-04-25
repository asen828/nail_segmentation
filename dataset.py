import cv2
import os
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
from torch.utils.tensorboard import SummaryWriter

#自定义指甲数据集
class NailDataset(Dataset):
    def __init__(self,root_dir,img_size=(256,256)):
        super().__init__()
        self.img_dir = os.path.join(root_dir,"images")      #数据地址
        self.label_dir = os.path.join(root_dir,"labels")    #标签地址
        self.img_size = img_size
        self.images = [     #将数据名加载到列表
            f for f in os.listdir(self.img_dir)
            if f.endswith((".jpg",".png","jpeg"))
        ]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir,img_name)  #合成文件路径

        base_name = os.path.splitext(img_name)[0]   #根据图片名，找对应的标签
        label_path = None
        for ext in [".png",".jpg",".jpeg",".gif"]:
            temp_path = os.path.join(self.label_dir,base_name + ext)
            if os.path.exists(temp_path):
               label_path = temp_path
               break
        if label_path is None:
           raise FileNotFoundError(f"找不到图片{img_name}对应的标签文件！！")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,self.img_size)     #缩放指定大小
        # image = cv2.GaussianBlur(image,(3,3),0)
        # cv2.imshow("image",image)
        image = torch.from_numpy(image).permute(2,0,1).float()/255.0  #numpy（H W C） -> tensor（C H W）
        

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)  #转成灰度图
        # label = cv2.adaptiveThreshold(label,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,0)   #二值化
        label = cv2.resize(label,self.img_size)
        # cv2.imshow("label",label)
        label = np.where(label>127,1.0,0.0)
        label = torch.from_numpy(label).unsqueeze(0).float()    #增加一个维度，与image的维度对齐
        # cv2.waitKey(0)

        return image,label
    


def get_dataloader(root_dir,bat_size,img_size,train_rate):
   
    NailSet = NailDataset(root_dir,img_size)
    
    # print(image.shape)
    # print(label.shape)
    total_size = len(NailSet)               #数据集总数
    train_size = int(total_size * train_rate)    #划分训练集和测试集数量
    test_size  = total_size - train_size

    print(f"数据集总数：{total_size}，训练集：{train_size},测试集：{test_size}")

    generator = torch.Generator().manual_seed(42) #随机划分，固定序列
    train_dataset,test_dataset = random_split(
        NailSet,[train_size,test_size],generator = generator
    )

    train_loader = DataLoader(train_dataset,batch_size=bat_size,shuffle=True,num_workers=0)
    test_loader  = DataLoader(test_dataset,batch_size=bat_size,shuffle=False,num_workers=0)

    return train_loader,test_loader

