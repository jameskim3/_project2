# Model
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import numpy as np
import os
class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained=True):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        num = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num, 4)

    def forward(self, image, targets):
        out = self.base_model(image)
        loss=nn.CrossEntropyLoss()(out,targets)
        return out, loss

if __name__ == "__main__":
    #model=SEResnext50_32x4d()
    #print(model)
    #config
    subject = 'Plant-pathology-2020'
    main_path = os.path.join("E:\\kaggle_imgs", subject)
    img_path = os.path.join(main_path, "images_224_224")
    data_path = os.path.join(main_path, "Data")
    save_path = os.path.join(main_path, "saved_models")
    best_path = save_path + "/200712"

    train_path=img_path
    df=pd.read_csv(data_path+"/train_fold.csv")

    #df
    df_train=df[df.fold!=0].reset_index(drop=True)
    df_valid=df[df.fold==0].reset_index(drop=True)

    print(df_train.CAT.value_counts())
    print(df_valid.CAT.value_counts())

