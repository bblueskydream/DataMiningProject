import copy

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.optim import lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, BertTokenizer, BertModel
from BertClassify import BertClassify
from CCFDataSet import CCFDataSet
import torch.nn.functional as F
import tensorflow as tf

# params
MODEL_PATH = "../data/bert-base-chinese"
DATA_PATH = "../data/train.json"
SAVE_PATH = "../model/model.pth"
TEST_PATH = "../data/testA.json"
FOLD = 10
MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-5
LR_MIN = 1e-6
WEIGHT_DECAY = 1e-6
CLASS_NUM = 36
EPOCH_TIMES = 40
T_MAX = 500


@torch.no_grad()
def test(bert_model, test_loader, device=DEVICE):
    bert_model.eval()
    # bert_model = resnet.resnet50()
    # bert_model =  bert_model.load_state_dict(torch.load(pth)['state_dict'])
    # bert_model.eval()
    
    classify_labels = []
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:
        ids = (torch.stack(tuple(data['input_ids']))).t().to(DEVICE, dtype=torch.long)
        mask = (torch.stack(tuple(data['attention_mask']))).t().to(DEVICE, dtype=torch.long)

        outputs = bert_model(ids, mask)
        print(outputs.size())
        batch_classify_labels = get_labels(outputs)
        classify_labels += batch_classify_labels
        print(len(classify_labels))

    return classify_labels


def get_labels(outputs):
    # pred_labels 和 true_labels 便于后续计算F1分数
    pred_labels = np.argmax(Tensor.cpu(outputs), axis=1)
    pred_labels = pred_labels.tolist()
    return pred_labels


model = BertModel.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)


# pth = r'Model\\model.pth'
# net = torch.load(pth, map_location=torch.device('cpu'))


# open testA
with open(TEST_PATH, "r", encoding="UTF-8") as f:
    file = f.readlines()

testFeature = pd.DataFrame(columns=['id', 'title', 'assignee', 'abstract', 'label_id'])
for data in file:
    json = eval(data)
    json['label_id'] = "0"
    # json = pd.DataFrame.from_dict(json, orient='index').T  # 将data从list转dataframe
    # testFeature = pd.concat([testFeature, json], ignore_index=True)  # 连接俩表
    # testFeature['label_id'] = "0"
    testFeature = testFeature.append(json, ignore_index=True)

testFeature['label_id'] = testFeature['label_id'].astype(int)
# testFeature['label_id'] = testFeature['label_id'].astype(int)
# testTensor = tf.constant(testFeature[['id', 'title', 'assignee', 'abstract', 'label_id']])

# testFeature = testFeature.head()
bert_model = BertClassify(model, config, CLASS_NUM)
bert_model.to(device=DEVICE)
bert_model.load_state_dict(torch.load(SAVE_PATH))
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# test
test_dataset = CCFDataSet(testFeature, tokenizer, MAX_LEN)
test_loader = DataLoader(dataset=test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)
models_results_dict = {}

print("test begin!")
preds = test(bert_model, test_loader)
models_results_dict["id"] = testFeature['id'].tolist()
models_results_dict["label"] = preds
    

test_result = pd.DataFrame(models_results_dict)
test_result.to_csv('../data/test_result.csv', index=False)
