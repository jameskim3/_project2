import os
import numpy as np
import pandas as pd
from sklearn import metrics

subject = 'H2'
main_path = os.path.join("E:\\kaggle_imgs", subject)
img_path = os.path.join(main_path, "images")
data_path = os.path.join(main_path, "Data")
save_path = os.path.join(main_path, "saved_models")
best_path = save_path+"/200712.bin"

def set_global():
    paths = [main_path, img_path, save_path, data_path]
    for fp in paths:
        print(fp)
        if not os.path.exists(fp):
            os.mkdir(fp)

#Model
import torch
import torch.nn as nn
import torchvision
import efficientnet_pytorch
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet,self).__init__()
        self.base_model=efficientnet_pytorch.EfficientNet.from_pretrained("efficientnet-b4")
        num=self.base_model._fc.in_features
        self.base_model._fc=nn.Linear(num,4,bias=True)
    def forward(self,image,targets):
        out=self.base_model(image)
        loss= nn.CrossEntropyLoss()(out,targets)
        return out,loss

#Augmentation
from albumentations import ( HorizontalFlip, IAAPerspective, ShiftScaleRotate, 
CLAHE, RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion, 
GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, 
MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine, IAASharpen, 
IAAEmboss, Flip, OneOf, Compose, Rotate, Cutout, HorizontalFlip, Normalize ) 
def get_aug(mode="train"):
    if mode =="train":
        aug=Compose([Rotate(15),
            OneOf([IAAAdditiveGaussianNoise(),GaussNoise(),], p=0.2),
            OneOf([MotionBlur(p=0.2),MedianBlur(blur_limit=3, p=0.1),Blur(blur_limit=3, p=0.1),], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([OpticalDistortion(p=0.3),GridDistortion(p=0.1),IAAPiecewiseAffine(p=0.3),], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
            Normalize(),
            ])
    else:
        aug=Compose([Normalize(),])

    return aug

#Dataset
train_bs=20
valid_bs=20
from MyLoader import ClassificationLoader
def get_dataset(df, mode="train",path=None):
    imgs=df.tar_path.values.tolist()
    if mode =="test":
        tar=np.zeros(len(imgs))
    else:
        tar=df.category.values

    aug=get_aug(mode)
    dataset=ClassificationLoader(
        image_paths=imgs,targets=tar,resize=None,augmentations=aug
    )
    batch_size = train_bs if mode=="train" else valid_bs
    shuffle=True if mode=="train" else False
    loader=torch.utils.data.DataLoader(
        dataset=dataset,batch_size=batch_size,shuffle=shuffle, num_workers=4,
    )
    return loader,tar

#train
from MyUtils import EarlyStopping
from MyEngine import Engine
def train(fold):
    df=pd.read_csv(data_path+"/train_fold.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    epochs=5

    df_train=df[df.fold!=fold].reset_index(drop=True)
    df_valid=df[df.fold==fold].reset_index(drop=True)

    path=img_path
    train_loader,train_tar=get_dataset(df_train,"train",path=path)
    valid_loader,valid_tar=get_dataset(df_valid,"valid",path=path)

    model=EfficientNet()
    model=model.to(device)
    model_save_path=save_path+f"/200801_fold_{fold}.bin"

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=0.001,mode="max",
    )
    es=EarlyStopping(patience=3,mode="max")

    for epoch in range(epochs):
        train_loss=Engine.train(train_loader, model, optimizer, device)
        preds,valid_loss=Engine.evaluate(valid_loader,model, device)
        preds=np.vstack(preds)
        print("pred",preds.shape)

        acc=np.array(preds.argmax(axis=1)==valid_tar).sum()/len(valid_tar)
        print(f"Epoch:{epoch}, acc: {acc}")
        scheduler.step(acc)
        es(acc,model,model_path=model_save_path)
        if es.early_stop:
            print("early stop")
            break

def predict(fold):
    df=pd.read_csv(data_path+"/test_path.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"

    path=img_path+"/test/"
    test_loader,_ = get_dataset(df[0:20],"test",path=path)

    model=EfficientNet()
    model_save_path=save_path+f"/200801_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)

    preds=Engine.predict(test_loader,model, device)
    preds=np.vstack(preds).argmax(axis=1)
    return preds

if __name__ =="__main__":
    #set_global()
    # train(0)
    # train(1)
    # train(2)
    # train(3)
    # train(4)

    p1 = predict(0)
    p2 = predict(1)
    p3 = predict(2)
    p4 = predict(3)
    p5 = predict(4)

    p1=p1.reshape(-1,1)
    p2=p2.reshape(-1,1)
    p3=p3.reshape(-1,1)
    p4=p4.reshape(-1,1)
    p5=p5.reshape(-1,1)
    f=np.concatenate((p1,p2,p3,p4,p5),axis=1)
    print(f)
    f=np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=f)
    sample = pd.read_csv(data_path+"/test_path.csv")
    sample["target"]=f
    sample.to_csv(data_path+"/sub_200801.csv", index=False)
    