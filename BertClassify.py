import torch
from torch import nn


class BertClassify(nn.Module):
    def __init__(self, model, config, num_class):
        super(BertClassify, self).__init__()
        self.bert = model
        self.config = config
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.config.hidden_size, num_class)
        self.fc.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.fc.bias is not None:
            self.fc.bias.data.zero_()
        return

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = out[0][:, 0, :]
        out = self.dropout(out)
        outputs = self.fc(out)
        return outputs
