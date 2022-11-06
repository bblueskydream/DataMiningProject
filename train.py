import copy
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim import lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, BertTokenizer, BertModel
from BertClassify import BertClassify
from CCFDataSet import CCFDataSet


# params
MODEL_PATH = "../data/bert-base-chinese"
DATA_PATH = "../data/train.json"
SAVE_PATH = "../model/model.pth"
FOLD = 10
MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 4
DEVICE = torch.device("cuda:0")
LR = 1e-5
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-6
CLASS_NUM = 36
EPOCH_TIMES = 40
T_MAX = 500


tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertModel.from_pretrained(MODEL_PATH)

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
# print(len(trainSet))
# print(len(valiSet))

trainLoader = DataLoader(dataset=trainSet, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
valiLoader = DataLoader(dataset=valiSet, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

bert_model = BertClassify(model, config, CLASS_NUM)
bert_model.to(device=DEVICE)

criterion = nn.CrossEntropyLoss(reduction='mean')
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
        ids = (torch.stack(tuple(data['input_ids']))).t().to(DEVICE, dtype=torch.long)
        mask = (torch.stack(tuple(data['attention_mask']))).t().to(DEVICE, dtype=torch.long)
        labels = (torch.stack(tuple(data['label']))).t().to(DEVICE, dtype=torch.long)

        size_temp = ids.size(0)
        # print(ids.size())
        # print(mask.size())
        outputs = bert_model(ids, mask)

        print(outputs.size())
        print(labels.size())
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
