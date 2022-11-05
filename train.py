import pandas as pd
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")


# params
data_path = "../data/train.json"
fold = 10

# read train.json and use KFold
with open(data_path, "r", encoding="UTF-8") as f:
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

