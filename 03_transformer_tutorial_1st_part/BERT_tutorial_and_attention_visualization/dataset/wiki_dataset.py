from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle


class BERTDataset(Dataset):
    def __init__(self, corpus_path, word2idx_path, seq_len, hidden_dim=384, on_memory=True):
        # hidden dimension for positional encoding
        self.hidden_dim = hidden_dim
        # define path of dicts
        self.word2idx_path = word2idx_path
        # define max length
        self.seq_len = seq_len
        # load whole corpus at once or not
        self.on_memory = on_memory
        # directory of corpus dataset
        self.corpus_path = corpus_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5

        # 加载字典
        with open(word2idx_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        # 加载语料
        with open(corpus_path, "r", encoding="utf-8") as f:
            if not on_memory:
                # 如果不将数据集直接加载到内存, 则需先确定语料行数
                self.corpus_lines = 0
                for _ in tqdm.tqdm(f, desc="Loading Dataset"):
                    self.corpus_lines += 1

            if on_memory:
                # 将数据集全部加载到内存
                self.lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading Dataset")]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            # 如果不全部加载到内存, 首先打开语料
            self.file = open(corpus_path, "r", encoding="utf-8")
            # 然后再打开同样的语料, 用来抽取负样本
            self.random_file = open(corpus_path, "r", encoding="utf-8")
            # 下面是为了错位抽取负样本
            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)

        t1_random, t1_label = self.random_char(t1)
        t2_random, t2_label = self.random_char(t2)

        t1 = [self.cls_index] + t1_random + [self.sep_index]
        t2 = t2_random + [self.sep_index]

        t1_label = [self.pad_index] + t1_label + [self.pad_index]
        t2_label = t2_label + [self.pad_index]

        segment_label = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        output = {"bert_input": torch.tensor(bert_input),
                  "bert_label": torch.tensor(bert_label),
                  "segment_label": torch.tensor(segment_label),
                  "is_next": torch.tensor([is_next_label])}

        return output

    def tokenize_char(self, segments):
        return [self.word2idx.get(char, self.unk_index) for char in segments]

    def random_char(self, sentence):
        char_tokens_ = list(sentence)
        char_tokens = self.tokenize_char(char_tokens_)

        output_label = []
        for i, token in enumerate(char_tokens):
            prob = random.random()
            if prob < 0.30:
                prob /= 0.30
                output_label.append(char_tokens[i])
                # 80% randomly change token to mask token
                if prob < 0.8:
                    char_tokens[i] = self.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9:
                    char_tokens[i] = random.randrange(len(self.word2idx))
            else:
                output_label.append(0)
        return char_tokens, output_label


    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]["text1"], self.lines[item]["text2"]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding="utf-8")
                line = self.file.__next__()
            line = eval(line)
            t1, t2 = line["text1"], line["text2"]
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))]["text2"]

        line = self.random_file.__next__()
        if line is None:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding="utf-8")
            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return eval(line)["text2"]