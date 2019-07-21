from torch.utils.data import Dataset
import tqdm
import json
import torch
import random
import numpy as np
from sklearn.utils import shuffle
import re


class CLSDataset(Dataset):
    def __init__(self, corpus_path, word2idx, max_seq_len, data_regularization=False):

        self.data_regularization = data_regularization
        self.word2idx = word2idx
        # define max length
        self.max_seq_len = max_seq_len
        # directory of corpus dataset
        self.corpus_path = corpus_path
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4
        self.num_index = 5

        # 加载语料
        with open(corpus_path, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            self.lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading Dataset")]
            # 打乱顺序
            self.lines = shuffle(self.lines)
            # 获取数据长度(条数)
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 得到tokenize之后的文本和与之对应的情感分类
        text, label = self.get_text_and_label(item)

        if self.data_regularization:
            # 数据正则, 有10%的几率再次分句
            if random.random() < 0.1:
                split_spans = [i.span() for i in re.finditer("，|；|。|？|!", text)]
                if len(split_spans) != 0:
                    span_idx = random.randint(0, len(split_spans) - 1)
                    cut_position = split_spans[span_idx][1]
                    if random.random() < 0.5:
                        if len(text) - cut_position > 2:
                            text = text[cut_position:]
                        else:
                            text = text[:cut_position]
                    else:
                        if cut_position > 2:
                            text = text[:cut_position]
                        else:
                            text = text[cut_position:]


        text_input = self.tokenize_char(text)

        # 添加#CLS#和#SEP#特殊token
        text_input = [self.cls_index] + text_input + [self.sep_index]
        # 如果序列的长度超过self.max_seq_len限定的长度, 则截断
        text_input = text_input[:self.max_seq_len]

        output = {"text_input": torch.tensor(text_input),
                  "label": torch.tensor([label])}
        return output

    def get_text_and_label(self, item):
        # 获取文本和标记
        text = self.lines[item]["text"]
        label = self.lines[item]["label"]
        return text, label

    def tokenize_char(self, segments):
        return [self.word2idx.get(char, self.unk_index) for char in segments]