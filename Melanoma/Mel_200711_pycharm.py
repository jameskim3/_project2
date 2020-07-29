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
file_path = os.path.join(saved_path,"200621_")
file_best = os.path.join(saved_path, "200621__epoch_ 0_acc_92.00")
train_info_pkl = os.path.join(data_path, "train_folds.csv")

import torch.nn as nn
import pretrainedmodels

class SEResnext50_32x4d(nn.Module):
    def __init__(self,pretrained="imagenet"):
        super(SEResnext50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.se_resnext50_32x4d(pretrained=True)
        self.I0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        batch, _, _, _ = image.shape
        x=self.base_model.features(image)
a=10
b=[1,2,3,4]

c=1

