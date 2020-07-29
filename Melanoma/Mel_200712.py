img_rows = 224
img_cols = 224
color_type = 3
batch_size = 48
epochs = 50

img_rows = 224
img_cols = 224
color_type = 3
batch_size = 48
epochs = 300

import os
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn import metrics

subject = 'Melanoma'
main_path = os.path.join("E:\\kaggle_imgs", subject)
img_path = os.path.join(main_path, "images")
data_path = os.path.join(main_path, "Data")
saved_path = os.path.join(main_path, "saved_models")
paths = [main_path, img_path, saved_path, data_path]
for fp in paths:
    print(fp)
    if not os.path.exists(fp):
        os.mkdir(fp)
file_path = os.path.join(saved_path, "200621_")
file_best = os.path.join(saved_path, "200621__epoch_ 0_acc_92.00")
train_info_pkl = os.path.join(data_path, "train_folds.csv")

import torch.nn as nn
import pretrainedmodels
import torchvision


## Model
class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained=True):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        num = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num, 1)

    def forward(self, image, targets):
        batch, _, _, _ = image.shape
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss


# Dataset
from torch.utils.data import Dataset
from PIL import Image

# train
import albumentations
import torch
from wtfml.engine import Engine
from wtfml.utils import EarlyStopping
from wtfml.data_loaders.image import ClassificationLoader
def train(fold):
    training_data_path = img_path+"/train3/"
    df = pd.read_csv(train_info_pkl)
    device = "cuda"
    epochs = 1#50
    train_bs = 32
    valid_bs = 16

    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    model = SEResnext50_32x4d(pretrained=True)
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=0
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")

    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=saved_path+f"/model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break

def predict(fold):
    #df load
    test_data_path=img_path+"/test3/"
    df=pd.read_csv(data_path+"/test.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model_path=saved_path+f"/model_fold_{fold}.bin"
    test_bs=32

    #augmentation
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    aug=albumentations.Compose([
        albumentations.Normalize(mean,std,max_pixel_value=255.0,always_apply=True)
    ])

    #dataset
    images=df.image_name.values.tolist()
    images=[test_data_path+file+".jpg" for file in images]
    targets=np.zeros(len(images))
    test_dataset=ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=aug,
    )

    #data loader
    test_loader=torch.utils.data.DataLoader(
        test_dataset,batch_size=test_bs,shuffle=False,num_workers=4)

    #model
    model=SEResnext50_32x4d(pretrained=True)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    predictions=Engine.predict(test_loader,model,device)
    predictions=np.vstack((predictions)).ravel()

    return predictions

if __name__ == "__main__":
    train(0)
    predict(0)