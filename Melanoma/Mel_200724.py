import os
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

subject="Melanoma"
main_path = os.path.join("E:\\kaggle_imgs", subject)
img_path = os.path.join(main_path, "images")
data_path = os.path.join(main_path, "Data")
save_path = os.path.join(main_path, "saved_models")
best_path = save_path+"/200712.bin"

train_bs=48
valid_bs=48

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
class Resnext_32x4d(nn.Module):
    def __init__(self,pretrained=True):
        super(Resnext_32x4d,self).__init__()
        self.base_model=torchvision.models.resnext50_32x4d(pretrained=pretrained)
        num=self.base_model.fc.in_features
        self.base_model.fc=nn.Linear(num,1)
    def forward(self,image,targets):
        out=self.base_model(image)
        loss=nn.BCEWithLogitsLoss()(out,targets.view(-1,1).type_as(out))
        return out,loss

#Augmentation
import albumentations
from MyLoader import ClassificationLoader
def get_aug(mode="train"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if mode =="train":
        aug=albumentations.Compose([
            albumentations.Normalize(mean, std,max_pixel_value=255.0,always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=15),
            albumentations.VerticalFlip(),albumentations.HorizontalFlip()
        ])
    else:
        aug=albumentations.Compose([
            albumentations.Normalize(mean, std,max_pixel_value=255.0,always_apply=True),
        ])
    return aug

#Dataset
def get_dataset(df,mode="train",path=None):
    imgs=df.image_name.values.tolist()
    imgs=[path+file+".jpg" for file in   imgs]
    if mode =="test":
        tar=np.zeros(len(imgs))
    else:
        tar=df.target.values

    aug=get_aug(mode)
    dataset=ClassificationLoader(
        image_paths=  imgs,targets= tar,resize=None,augmentations=aug
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
    df=pd.read_csv(data_path+"/df_train_fold.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    epochs=1

    df_train=df[df.fold!=fold].reset_index(drop=True)
    df_valid=df[df.fold==fold].reset_index(drop=True)

    path=img_path+"/train3/"
    train_loader,train_tar=get_dataset(df_train,"train",path=path)
    valid_loader,valid_tar=get_dataset(df_valid,"valid",path=path)

    model=Resnext_32x4d(pretrained=True)
    model=model.to(device)
    model_save_path=save_path+f"/200724_fold_{fold}.bin"

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,patience=3,threshold=0.001,mode="max",
    )
    es=EarlyStopping(patience=3,mode="max")

    for epoch in range(epochs):
        train_loss=Engine.train(train_loader, model, optimizer, device)
        preds,valid_loss=Engine.evaluate(valid_loader,model, device)
        preds=np.vstack(preds).flatten()
        auc=metrics.roc_auc_score(valid_tar,preds)
        print(f"Epoch:{epoch}, AUC: {auc}")
        scheduler.step(auc)
        es(auc,model,model_path=model_save_path)
        if es.early_stop:
            print("early stop")
            break
def predict(fold):
    df=pd.read_csv(data_path+"/test.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"

    path=img_path+"/test3/"
    test_loader,_ = get_dataset(df,"test",path=path)

    model=Resnext_32x4d(pretrained=True)
    model_save_path=save_path+f"/200724_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_save_path))
    model=model.to(device)

    preds=Engine.predict(test_loader,model, device)
    preds=np.vstack(preds).flatten()
    return preds

if __name__ == "__main__":
    # set_global()
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

    predictions = (p1 + p2 + p3 + p4 + p5) / 5
    sample = pd.read_csv(data_path+"/sample_submission.csv")
    sample.loc[:, "target"] = predictions
    sample.to_csv("sub_200725.csv", index=False)

