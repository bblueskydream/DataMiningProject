from torch.utils.data import Dataset


class CCFDataSet(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.id = df["id"]
        self.title = df["title"]
        self.assignee = df["assignee"]
        self.abstract = df["abstract"]
        self.label_id = df["label_id"]
        return

    def __getitem__(self, index):
        data_id = self.id[index]
        title = self.title[index]
        assignee = self.assignee[index]
        abstract = self.abstract[index]
        label = self.label_id[index]
        text = "专利标题是《{}》，属于“{}”公司，摘要如下：{}".format(title, assignee, abstract)
        inputs = self.tokenizer.encode_plus(text, truncation=True, add_special_tokens=True, max_length=self.max_len)

        return {'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'label': label}

    def __len__(self):
        return len(self.df)


