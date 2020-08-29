# config

import os
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn import metrics

subject = 'Melanoma'
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
class SEResnext50_32x4d(nn.Module):
    def __init__(self,pretrained=True):
        super(SEResnext50_32x4d,self).__init__()
        self.base_model=torchvision.models.resnext50_32x4d(pretrained=pretrained)
        num=self.base_model.fc.in_features
        self.base_model.fc=nn.Linear(num,1)
    def forward(self,image,targets):
        out=self.base_model(image)
        loss=nn.BCEWithLogitsLoss()(out,targets.view(-1,1).type_as(out))
        return out,loss

import albumentations
# from wtfml.data_loaders.image import ClassificationLoader
# from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from MyLoader import ClassificationLoader
from MyEngine import Engine
from MyUtils import EarlyStopping
def train(fold):
    #config
    train_path=img_path+"/train3/"
    df=pd.read_csv(data_path+"/train_folds.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    epochs=1
    train_bs=32
    valid_bs=32

    #df
    df_train=df[df.fold!=fold].reset_index(drop=True)
    df_valid=df[df.fold==fold].reset_index(drop=True)

     #Aug
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug=albumentations.Compose([
        albumentations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True),
        albumentations.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=15),
        albumentations.VerticalFlip(p=0.5),albumentations.HorizontalFlip(p=0.5),
    ])
    valid_aug=albumentations.Compose([
        albumentations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True),
    ])

    #Dataset
    train_imgs=df_train.image_name.values.tolist()
    train_imgs=[img_path+"/train3/"+ file +".jpg" for file in train_imgs]
    train_targets=df_train.target.values

    valid_imgs = df_valid.image_name.values.tolist()
    valid_imgs = [img_path + "/train3/" + file+".jpg"  for file in valid_imgs]
    valid_targets = df_valid.target.values

    train_dataset=ClassificationLoader(
        image_paths=train_imgs,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )
    valid_dataset=ClassificationLoader(
        image_paths=valid_imgs,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )
    train_loader=torch.utils.data.DataLoader(
        train_dataset,batch_size=train_bs,shuffle=True,num_workers=4
    )
    valid_loader=torch.utils.data.DataLoader(
        valid_dataset,batch_size=valid_bs,shuffle=True,num_workers=4
    )

    # model
    model = SEResnext50_32x4d(pretrained=True)
    model = model.to(device)
    model_path=save_path+f"/200712_fold_{fold}.bin"
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )
    es=EarlyStopping(patience=3,mode="max")

    for epoch in range(epochs):
        train_loss=Engine.train(train_loader,model, optimizer,device=device)
        preds,valid_loss=Engine.evaluate(valid_loader, model, device=device)
        preds=np.vstack(preds).ravel()
        auc=metrics.roc_auc_score(valid_targets,preds)
        print(f"Epoch {epoch}, AUC {auc}")
        scheduler.step(auc)
        es(auc,model,model_path=model_path)
        if es.early_stop:
            print("early stop")
            break

def predict(fold):
    #config : path, df, device, epochs, train_bs
    test_path=img_path+"/test3/"
    df=pd.read_csv(data_path+"/test.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    epochs=1
    test_bs=18
    df_test=df

    #Aug : mean, std, train_aug,valid_aug
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug=albumentations.Compose([
        albumentations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True)
    ])

    #Dataset: train_imgs, train_targets, train_dataset, train_loader
    test_imgs=df.image_name.values.tolist()
    test_imgs=[test_path+file+".jpg" for file in test_imgs]
    print(test_imgs[0])
    test_target=np.zeros(len(test_imgs))
    test_dataset=ClassificationLoader(
        image_paths=test_imgs,
        targets=test_target,
        resize=None,
        augmentations=aug
    )
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=test_bs,shuffle=False,num_workers=4
    )

    #Model Utils: model, optimizer, scheduler, es
    model=SEResnext50_32x4d(pretrained=True)
    model_path=save_path+f"/200712_fold_{fold}.bin"
    model.load_state_dict(torch.load(model_path))
    model=model.to(device)

    # train, predict
    preds=Engine.predict(test_loader,model,device=device)
    preds=np.vstack(preds).flatten()

    return preds

if __name__ == "__main__":
    set_global()
    #train(1)
    predict(1)
