from torch.utils.data import Dataset


class CCFDataSet(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

        return

    def __getitem__(self, index):
        return

    def __len__(self):
        return len(self.df)


