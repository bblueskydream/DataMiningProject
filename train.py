import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForMaskedLM
from CCFDataSet import CCFDataSet
from CCFModel import CCFNet

# params
MODEL_PATH = "./data/bert-base-chinese"
DATA_PATH = "./data/train.json"
FOLD = 10
MAX_LEN = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda:0")
LR = 1e-5
LR_MIN = 1e-6
# SEED = 30
CATEGORY_NUM = 36


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

# read train.json and use KFold
with open(DATA_PATH, "r", encoding="UTF-8") as f:
    file = f.readlines()

df = pd.DataFrame(columns=['id', 'title', 'assignee', 'abstract', 'label_id'])
for data in file:
    json = eval(data)
    df = df.append(json, ignore_index=True)

df['label_id'] = df['label_id'].astype(int)

sfolder = StratifiedKFold(n_splits=10)

for f, (_, val_) in enumerate(sfolder.split(X=df, y=df.label_id)):
    df.loc[val_, "kfold"] = int(f)

df["kfold"] = df["kfold"].astype(int)
df.groupby("kfold")['label_id'].value_counts()
# df.to_csv("df.csv")

# train: 8  test: 2

selectFold0 = 0
selectFold1 = 1
train = df[df["kfold"] != selectFold0 & df["kfold"] != selectFold1].reset_index(drop=True)
test = df[df["kfold"] == selectFold0 | df["kfold"] == selectFold1].reset_index(drop=True)

trainSet = CCFDataSet(train, tokenizer, MAX_LEN)
testSet = CCFDataSet(test, tokenizer, MAX_LEN)

trainLoader = DataLoader(trainSet, )
