# config

import os
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
from sklearn import metrics

subject='Plant-pathology-2020'
main_path = os.path.join("E:\\kaggle_imgs", subject)
img_path = os.path.join(main_path, "images_224_224")
data_path = os.path.join(main_path, "Data")
save_path = os.path.join(main_path, "saved_models")
best_path = save_path+"/200712"

epochs = 100
train_bs = 32
valid_bs = 16

def set_global():
    paths = [main_path, img_path, save_path, data_path]
    for fp in paths:
        print(fp)
        if not os.path.exists(fp):
            os.mkdir(fp)

def make_csv():
    df_train = pd.read_csv(data_path + "/train.csv")
    df_train["CAT"] = df_train[['healthy', 'multiple_diseases', 'rust', 'scab']].values.argmax(axis=1)
    df_train["id"] = df_train["image_id"].apply(lambda x: int(x.split("_")[1]))
    X = df_train['id'].values
    y = df_train['CAT'].values
    skf = model_selection.StratifiedKFold(n_splits=5, random_state=22)
    df_train["fold"] = -1
    sum = 0
    for i, (trn_idx, vld_idx) in enumerate(skf.split(X, y)):
        df_train.loc[vld_idx, "fold"] = i

    df_train.to_csv(data_path+"/train_fold.csv", index=False)

# Model
import torch
import torch.nn as nn
import torchvision
class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained=True):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        num = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num, 4)

    def forward(self, image):
        out = self.base_model(image)
        return out

from albumentations import ( HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE,
                             RandomRotate90, Transpose, ShiftScaleRotate, Blur, OpticalDistortion,
                             GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise,
                             GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast,
                             IAAPiecewiseAffine, IAASharpen, IAAEmboss, Flip, OneOf, Compose,
                             Rotate, Cutout, HorizontalFlip, Normalize )
from albumentations.pytorch import ToTensor
train_aug = Compose([
    Rotate(15),
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.2),
    OneOf([
        MotionBlur(p=0.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    OneOf([
        OpticalDistortion(p=0.3),
        GridDistortion(p=0.1),
        IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    OneOf([
        CLAHE(clip_limit=2),
        IAASharpen(),
        IAAEmboss(),
        RandomBrightnessContrast(),
    ], p=0.3),
    HueSaturationValue(p=0.3),
    Normalize(),
])
valid_aug = Compose([
    Normalize(),
])

from torch.utils.data import Dataset
from PIL import Image
class MyDataset(Dataset):
    def __init__(self, image_paths, targets, resize, augmentations):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        targets = self.targets[idx]
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        image = self.augmentations(image=image)["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }

def get_images_by_fold(df, fold):
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    train_imgs = df_train.image_id.values.tolist()
    train_imgs = [img_path + "/" + file + ".png" for file in train_imgs]
    train_targets = df_train.CAT.values

    valid_imgs = df_valid.image_id.values.tolist()
    valid_imgs = [img_path + "/" + file + ".png" for file in valid_imgs]
    valid_targets = df_valid.CAT.values

    train_dataset = MyDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )
    valid_dataset = MyDataset(
        image_paths=valid_imgs,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=True, num_workers=4
    )

    return train_loader,valid_loader

def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    device="cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        tk0 = tqdm(test_set, total=len(test_set), disable=False)
        for i, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)

            outputs = net(data["image"])

            _, predicted = torch.max(outputs.data, 1)

            total += data["targets"].size(0)
            correct += (predicted == data["targets"]).sum().item()

    acc = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    return acc

from tqdm import tqdm
from wtfml.utils import AverageMeter
from wtfml.utils import EarlyStopping
def train(fold):
    #config
    df=pd.read_csv(data_path+"/train_fold.csv")
    device="cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = SEResnext50_32x4d(pretrained=True)
    model = model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    es=EarlyStopping(patience=5, mode="max")
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times
        losses = AverageMeter()
        # scheduler.step()

        for my_iter in range(fold,fold+1,1):
            train_loader, valid_loader = get_images_by_fold(df, my_iter)
            print("iter %d/5 load complete" % my_iter)
            tk0 = tqdm(train_loader, total=len(train_loader), disable=False)
            for i, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(data["image"])
                loss = criterion(outputs, data["targets"])
                loss.backward()
                optimizer.step()

                #loss update
                losses.update(loss.item(), train_loader.batch_size)
                tk0.set_postfix(loss=losses.avg)

        # Check Accuracy
        acc = acc_check(model, valid_loader, epoch)
        scheduler.step(acc)
        es(acc,model,model_path=save_path+f"/200712_acc_{acc}.bin")
        if es.early_stop:
            print("early stop")
            break

if __name__ == "__main__":
    # set_global()
    # make_csv()
    train(0)