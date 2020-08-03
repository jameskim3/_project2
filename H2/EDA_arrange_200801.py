import os
import shutil
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics

src_path=os.path.join("e:/kaggle_imgs/Plant-pathology-2020/images_224_224/")
tar_path=os.path.join("e:/kaggle_imgs/H2/images/")
src_train_path=os.path.join("e:/kaggle_imgs/Plant-pathology-2020/data","train.csv")
tar_bin=["0_Normal","1_OverProcess","2_UnderProcess","3_Defect"]

def arrange_files():
    files=os.listdir(src_path)
    df_source=pd.DataFrame({"images":files})
    df_train=pd.read_csv(src_train_path)
    df_train["Category"]=df_train[["healthy","multiple_diseases","rust","scab"]].values.argmax(axis=1)
    df_train["tar_folder"]=df_train.Category.apply(lambda x:tar_bin[x])
    for (src,tar) in zip(df_train.image_id,df_train.tar_folder):
        src_fp=src_path+src+".png"
        tar_fp=tar_path+tar+"/"+src+".png"
        shutil.copy(src_fp,tar_fp)

    return df_train

def make_fold():
    cat_files=[]
    cat_values=[]
    for i in range(4):
        files=np.array(os.listdir(tar_path+tar_bin[i])).reshape(-1,1)
        val=np.ones(len(files)).reshape(-1,1)*i
        cat_files.append(files)
        cat_values.append(val)
    images=np.vstack(cat_files)
    cats=np.vstack(cat_values)
    df_train=pd.DataFrame({"images":images.flatten(),"category":cats.flatten()})
    df_train.category=df_train.category.apply(lambda x:int(x))
    X=df_train.sample(frac=1).values
    y=df_train.category.values
    skf=model_selection.StratifiedKFold(n_splits=5,random_state=5)
    df_train["fold"]=-1
    for i,(_,val_idx) in enumerate(skf.split(X,y)):
        df_train.loc[val_idx,"fold"]=i
    
    df_train["tar_path"]=df_train.category.apply(lambda x:tar_bin[x])
    df_train["tar_path"]=tar_path+df_train.tar_path+"/"+df_train.images

    df_train.to_csv("e:/kaggle_imgs/H2/Data/train_fold.csv",index=False)
    for i in range(5):
        print(f"Case[{i}]",df_train.category[df_train.fold==i].value_counts())
    return df_train

def make_test():
    df_test=pd.read_csv("E:/kaggle_imgs/H2/Data/test.csv")
    df_test["tar_path"]=tar_path+"test/"+df_test.image_id+".png"
    df_test.to_csv("e:/kaggle_imgs/H2/Data/test_path.csv",index=False)

    return df_test

if __name__ == "__main__":
    #df_train=arrange_files()
    #df_train=make_fold()
    #df_test=make_test()
    pass
    