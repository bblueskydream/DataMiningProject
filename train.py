import copy
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from BertClassify import BertClassify
from CCFDataSet import CCFDataSet


def criterion(output, label):
    return nn.CrossEntropyLoss()(output, label)


# params
MODEL_PATH = "../data/bert-base-chinese"
DATA_PATH = "../data/train.json"
SAVE_PATH = "../model/model.pth"
FOLD = 10
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
DEVICE = torch.device("cuda:0")
LR = 1e-5
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-6
CLASS_NUM = 36
EPOCH_TIMES = 40
T_MAX = 500


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)

config = AutoConfig.from_pretrained(MODEL_PATH)

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

# train: 9  vali: 1

selectFold0 = 0
# train = df[(df["kfold"] != selectFold0)].reset_index(drop=True)
# test = df[(df["kfold"] == selectFold0)].reset_index(drop=True)
train = df.reset_index(drop=True)
vali = df.reset_index(drop=True)

trainSet = CCFDataSet(train, tokenizer, MAX_LEN)
valiSet = CCFDataSet(vali, tokenizer, MAX_LEN)
print(len(trainSet))
print(len(valiSet))

trainLoader = DataLoader(dataset=trainSet, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
valiLoader = DataLoader(dataset=valiSet, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

bert_model = BertClassify(model, config, CLASS_NUM)
bert_model.to(device=DEVICE)

optimizer = AdamW(bert_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=LR_MIN)

best_loss = 999999
best_weights = copy.deepcopy(bert_model.state_dict())
# train
for epoch in range(EPOCH_TIMES):
    bert_model.train()
    size = 0
    loss = 0.0

    # train
    bar = tqdm(enumerate(trainLoader), total=len(trainLoader))
    for step, data in bar:
        print(data)
        data1 = dict()
        # data1['input_ids'] = [x for x in data['input_ids']]
        # data1['attention_mask'] = [x for x in data['attention_mask']]
        # data1['label'] = [x for x in data['label']]
        # ids = torch.tensor(data1['input_ids']).to(DEVICE, dtype=torch.long)
        # mask = torch.tensor(data1['attention_mask']).to(DEVICE, dtype=torch.long)
        # labels = torch.tensor(data1['label']).to(DEVICE, dtype=torch.long)
        ids = (torch.cat(tuple(data['input_ids']))).to(DEVICE, dtype=torch.long)
        mask = (torch.cat(tuple(data['attention_mask']))).to(DEVICE, dtype=torch.long)
        labels = (torch.cat(tuple(data['label']))).to(DEVICE, dtype=torch.long)

        size_temp = ids.size(0)
        outputs = bert_model(ids, mask)

        loss_temp = criterion(outputs, labels)
        loss_temp.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
        loss += (loss_temp * size_temp)
        step += size_temp

        epoch_loss = loss / size
        bar.set_postfix(Epoch=epoch, Tran_loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    if loss < best_loss:
        print(f'loss improved {best_loss} --> {loss}')
        best_loss = loss
        best_weights = copy.deepcopy(bert_model.state_dict())
        torch.save(bert_model.state_dict(), SAVE_PATH)

print(f'best loss is {best_loss}')
print(f'best weights is {best_weights}')
