import torch
from models.bert_model import BertModel, BertConfig
from dataset.inference_dataloader import preprocessing
import warnings
import json
import math
import os
import configparser
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='./SimHei.ttf')

class Plot_Attention:
    def __init__(self,
                 max_seq_len,
                 batch_size=1,
                 with_cuda=False,
                 ):
        # 加载配置文件
        config_ = configparser.ConfigParser()
        config_.read("./model_config.ini")
        self.config = config_["DEFAULT"]
        # 词量, 注意在这里实际字(词)汇量 = vocab_size - 20,
        # 因为前20个token用来做一些特殊功能, 如padding等等
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        # 是否使用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 限定的单句最大长度
        self.max_seq_len = max_seq_len
        # 初始化超参数的配置
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化bert模型
        self.bert_model = BertModel(config=bertconfig)
        self.bert_model.to(self.device)
        # 加载字典
        self.word2idx = self.load_dic(self.config["word2idx_path"])
        # 初始化预处理器
        self.process_batch = preprocessing(hidden_dim=bertconfig.hidden_size,
                                           max_positions=max_seq_len,
                                           word2idx=self.word2idx)
        # 加载BERT预训练模型
        self.load_model(self.bert_model, dir_path=self.config["state_dict_dir"])
        # disable dropout layers
        self.bert_model.eval()

    def load_dic(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_model(self, model, dir_path="./output"):
        # 加载模型
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # 不加载masked language model 和 next sentence 的层和参数
        checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                          if k[:4] == "bert"}
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded for evaluation!".format(checkpoint_dir))


    def get_attention(self, text_list, batch_size=1):
        """
        :param text_list:
        :param batch_size: 为了注意力矩阵的可视化, batch_size只能为1, 即单句
        :return:
        """
        # 异常判断
        if isinstance(text_list, str):
            text_list = [text_list, ]
        len_ = len(text_list)
        text_list = [i for i in text_list if len(i) != 0]
        if len(text_list) == 0:
            raise NotImplementedError("输入的文本全部为空, 长度为0!")
        if len(text_list) < len_:
            warnings.warn("输入的文本中有长度为0的句子, 它们将被忽略掉!")

        # max_seq_len=self.max_seq_len+2 因为要留出cls和sep的位置
        max_seq_len = max([len(i) for i in text_list])
        # 预处理, 获取batch
        texts_tokens, positional_enc = \
            self.process_batch(text_list, max_seq_len=max_seq_len)
        # 准备positional encoding
        positional_enc = torch.unsqueeze(positional_enc, dim=0).to(self.device)

        # 正向
        n_batches = math.ceil(len(texts_tokens) / batch_size)

        # 数据按mini batch切片过正向, 这里为了可视化所以吧batch size设为1
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            # 切片
            texts_tokens_ = texts_tokens[start: end].to(self.device)
            attention_matrices = self.bert_model.forward(input_ids=texts_tokens_,
                                                         positional_enc=positional_enc,
                                                         get_attention_matrices=True)
            # 因为batch size=1所以直接返回每层的注意力矩阵
            return [i.detach().numpy() for i in attention_matrices]

    def __call__(self, text, layer_num, head_num):
        # 获取attention矩阵的list
        attention_matrices = self.get_attention(text)
        # 准备热图的tags, 把字分开
        labels = [i + " " for i in list(text)]
        labels = ["#CLS# ", ] + labels + ["#SEP# ", ]
        # 画图
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_matrices[layer_num][0][head_num])
        plt.yticks(range(len(labels)), labels, fontproperties=font, fontsize=18)
        plt.xticks(range(len(labels)), labels, fontproperties=font, fontsize=18)
        # plt.show()

    def find_most_recent_state_dict(self, dir_path):
        # 找到模型存储的最新的state_dict路径
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]


if __name__ == '__main__':
    model = Plot_Attention(max_seq_len=256)
    model("测试一下", layer_num=3, head_num=2)