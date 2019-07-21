from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.sentiment_dataset_v2 import CLSDataset
from models.bert_sentiment_analysis_v2 import *
from sklearn import metrics
from metrics import *

import tqdm
import pandas as pd
import numpy as np
import configparser
import os
import json


class Sentiment_trainer:
    def __init__(self, max_seq_len,
                 batch_size,
                 lr, # 学习率
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):
        config_ = configparser.ConfigParser()
        config_.read("./config/sentiment_model_config.ini")
        self.config = config_["DEFAULT"]
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        self.lr = lr
        # 加载字典
        with open(self.config["word2idx_path"], "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        # 允许的最大序列长度
        self.max_seq_len = max_seq_len
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=self.vocab_size)
        # 初始化BERT情感分析模型
        self.bert_model = Bert_Sentiment_Analysis(config=bertconfig)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明训练数据集, 按照pytorch的要求定义数据集class
        train_dataset = CLSDataset(corpus_path=self.config["train_corpus_path"],
                                   word2idx=self.word2idx,
                                   max_seq_len=self.max_seq_len,
                                   data_regularization=True
                                   )
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           collate_fn=lambda x: x # 这里为了动态padding
                                           )
        # 声明测试数据集
        test_dataset = CLSDataset(corpus_path=self.config["test_corpus_path"],
                                  word2idx=self.word2idx,
                                  max_seq_len=self.max_seq_len,
                                  data_regularization=False
                                  )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          collate_fn=lambda x: x)
        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        # 声明需要优化的参数, 并传入Adam优化器
        self.optim_parameters = list(self.bert_model.parameters())

        # all_parameters = list(self.bert_model.named_parameters())
        # lis_ = ["dense.weight", "dense.bias", "final_dense.weight", "final_dense.bias"]
        # # self.optim_parameters = [i[1] for i in all_parameters if i[0] in lis_]
        # self.optim_parameters = list(self.bert_model.parameters())

        self.init_optimizer(lr=self.lr)
        if not os.path.exists(self.config["state_dict_dir"]):
            os.mkdir(self.config["state_dict_dir"])

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc


    def load_model(self, model, dir_path="../output", load_bert=False):
        checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        checkpoint = torch.load(checkpoint_dir)
        # 情感分析模型刚开始训练的时候, 需要载入预训练的BERT,
        # 这是我们不载入模型原本用于训练Next Sentence的pooler
        # 而是重新初始化了一个
        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(checkpoint_dir))

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        self.iteration(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        # 一个epoch的测试, 并返回测试集的auc
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.test_dataloader, train=False)

    def padding(self, output_dic_lis):
        """动态padding, 以当前mini batch内最大的句长进行补齐长度"""
        text_input = [i["text_input"] for i in output_dic_lis]
        text_input = torch.nn.utils.rnn.pad_sequence(text_input, batch_first=True)
        label = torch.cat([i["label"] for i in output_dic_lis])
        return {"text_input": text_input,
                "label": label}

    def iteration(self, epoch, data_loader, train=True, df_name="df_log.pickle"):
        # 初始化一个pandas DataFrame进行训练日志的存储
        df_path = self.config["state_dict_dir"] + "/" + df_name
        if not os.path.isfile(df_path):
            df = pd.DataFrame(columns=["epoch", "train_loss", "train_auc",
                                       "test_loss", "test_auc"
                                       ])
            df.to_pickle(df_path)
            print("log DataFrame created!")

        # 进度条显示
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0
        # 存储所有预测的结果和标记, 用来计算auc
        all_predictions, all_labels = [], []

        for i, data in data_iter:
            # padding
            data = self.padding(data)
            # 将数据发送到计算设备
            data = {key: value.to(self.device) for key, value in data.items()}
            # 根据padding之后文本序列的长度截取相应长度的位置编码,
            # 并发送到计算设备
            positional_enc = self.positional_enc[:, :data["text_input"].size()[-1], :].to(self.device)

            # 正向传播, 得到预测结果和loss
            predictions, loss = self.bert_model.forward(text_input=data["text_input"],
                                                        positional_enc=positional_enc,
                                                        labels=data["label"]
                                                        )
            # 提取预测的结果和标记, 并存到all_predictions, all_labels里
            # 用来计算auc
            predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
            labels = data["label"].cpu().numpy().reshape(-1).tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            # 计算auc
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                                                     y_score=all_predictions)
            auc = metrics.auc(fpr, tpr)

            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()

            if train:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": total_loss/(i+1), "train_auc": auc,
                    "test_loss": 0, "test_auc": 0
                }

            else:
                log_dic = {
                    "epoch": epoch,
                    "train_loss": 0, "train_auc": 0,
                    "test_loss": total_loss/(i+1), "test_auc": auc
                }

            if i % 10 == 0:
                data_iter.write(str({k: v for k, v in log_dic.items() if v != 0}))

        threshold_ = find_best_threshold(all_predictions, all_labels)
        print(str_code + " best threshold: " + str(threshold_))

        # 将当前epoch的情况记录到DataFrame里
        if train:
            df = pd.read_pickle(df_path)
            df = df.append([log_dic])
            df.reset_index(inplace=True, drop=True)
            df.to_pickle(df_path)
        else:
            log_dic = {k: v for k, v in log_dic.items() if v != 0 and k != "epoch"}
            df = pd.read_pickle(df_path)
            df.reset_index(inplace=True, drop=True)
            for k, v in log_dic.items():
                df.at[epoch, k] = v
            df.to_pickle(df_path)
            # 返回auc, 作为early stop的衡量标准
            return auc

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, state_dict_dir="../output", file_path="bert.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)
        save_path = state_dict_dir + "/" + file_path + ".epoch.{}".format(str(epoch))
        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)


if __name__ == '__main__':
    def init_trainer(dynamic_lr, batch_size=24):
        trainer = Sentiment_trainer(max_seq_len=300,
                                    batch_size=batch_size,
                                    lr=dynamic_lr,
                                    with_cuda=True,)
        return trainer, dynamic_lr

    start_epoch = 0
    train_epoches = 9999
    trainer, dynamic_lr = init_trainer(dynamic_lr=1e-06, batch_size=24)


    all_auc = []
    threshold = 999
    patient = 10
    best_loss = 999999999
    for epoch in range(start_epoch, start_epoch + train_epoches):
        if epoch == start_epoch and epoch == 0:
            # 第一个epoch的训练需要加载预训练的BERT模型
            trainer.load_model(trainer.bert_model, dir_path="./bert_state_dict", load_bert=True)
        elif epoch == start_epoch:
            trainer.load_model(trainer.bert_model, dir_path=trainer.config["state_dict_dir"])
        print("train with learning rate {}".format(str(dynamic_lr)))
        # 训练一个epoch
        trainer.train(epoch)
        # 保存当前epoch模型参数
        trainer.save_state_dict(trainer.bert_model, epoch,
                                state_dict_dir=trainer.config["state_dict_dir"],
                                file_path="sentiment.model")

        auc = trainer.test(epoch)

        all_auc.append(auc)
        best_auc = max(all_auc)
        if all_auc[-1] < best_auc:
            threshold += 1
            dynamic_lr *= 0.8
            trainer.init_optimizer(lr=dynamic_lr)
        else:
            # 如果
            threshold = 0

        if threshold >= patient:
            print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_auc))))
            print("early stop!")
            break