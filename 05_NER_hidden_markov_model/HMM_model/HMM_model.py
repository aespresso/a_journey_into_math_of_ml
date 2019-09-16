import numpy as np
from utils import *
from tqdm import tqdm


class HMM_NER:
    def __init__(self, char2idx_path, tag2idx_path):
        # 载入一些字典
        # char2idx: 字 转换为 token
        self.char2idx = load_dict(char2idx_path)
        # tag2idx: 标签转换为 token
        self.tag2idx = load_dict(tag2idx_path)
        # idx2tag: token转换为标签
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}
        # 初始化隐状态数量(实体标签数)和观测数量(字数)
        self.tag_size = len(self.tag2idx)
        self.vocab_size = max([v for _, v in self.char2idx.items()]) + 1
        # 初始化A, B, pi为全0
        self.transition = np.zeros([self.tag_size,
                                    self.tag_size])
        self.emission = np.zeros([self.tag_size,
                                  self.vocab_size])
        self.pi = np.zeros(self.tag_size)
        # 偏置, 用来防止log(0)或乘0的情况
        self.epsilon = 1e-8

    def fit(self, train_dic_path):
        """
        fit用来训练HMM模型
        :param train_dic_path: 训练数据目录
        """
        print("initialize training...")
        train_dic = load_data(train_dic_path)
        # 估计转移概率矩阵, 发射概率矩阵和初始概率矩阵的参数
        self.estimate_transition_and_initial_probs(train_dic)
        self.estimate_emission_probs(train_dic)
        # take the logarithm
        # 取log防止计算结果下溢
        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)
        print("DONE!")


    def estimate_emission_probs(self, train_dic):
        """
        发射矩阵参数的估计
        estimate p( Observation | Hidden_state )
        :param train_dic:
        :return:
        """
        print("estimating emission probabilities...")
        for dic in tqdm(train_dic):
            for char, tag in zip(dic["text"], dic["label"]):
                self.emission[self.tag2idx[tag],
                              self.char2idx[char]] += 1
        self.emission[self.emission == 0] = self.epsilon
        self.emission /= np.sum(self.emission, axis=1, keepdims=True)


    def estimate_transition_and_initial_probs(self, train_dic):
        """
        转移矩阵和初始概率的参数估计, 也就是bigram二元模型
        estimate p( Y_t+1 | Y_t )
        :param train_dic:
        :return:
        """
        print("estimating transition and initial probabilities...")
        for dic in tqdm(train_dic):
            for i, tag in enumerate(dic["label"][:-1]):
                if i == 0:
                    self.pi[self.tag2idx[tag]] += 1
                curr_tag = self.tag2idx[tag]
                next_tag = self.tag2idx[dic["label"][i+1]]
                self.transition[curr_tag, next_tag] += 1
        self.transition[self.transition == 0] = self.epsilon
        self.transition /= np.sum(self.transition, axis=1, keepdims=True)
        self.pi[self.pi == 0] = self.epsilon
        self.pi /= np.sum(self.pi)

    def get_p_Obs_State(self, char):
        # 计算p( observation | state)
        # 如果当前字属于未知, 则讲p( observation | state)设为均匀分布
        char_token = self.char2idx.get(char, 0)
        if char_token == 0:
            return np.log(np.ones(self.tag_size)/self.tag_size)
        return np.ravel(self.emission[:, char_token])

    def predict(self, text):
        # 预测并打印出预测结果
        # 维特比算法解码
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tag_id = self.viterbi_decode(text)
        self.print_func(text, best_tag_id)

    def print_func(self, text, best_tags_id):
        # 用来打印预测结果
        for char, tag_id in zip(text, best_tags_id):
            print(char+"_"+self.idx2tag[tag_id]+"|", end="")

    def viterbi_decode(self, text):
        """
        维特比解码, 详见视频教程或文字版教程
        :param text: 一段文本string
        :return: 最可能的隐状态路径
        """
        # 得到序列长度
        seq_len = len(text)
        # 初始化T1和T2表格
        T1_table = np.zeros([seq_len, self.tag_size])
        T2_table = np.zeros([seq_len, self.tag_size])
        # 得到第1时刻的发射概率
        start_p_Obs_State = self.get_p_Obs_State(text[0])
        # 计算第一步初始概率, 填入表中
        T1_table[0, :] = self.pi + start_p_Obs_State
        T2_table[0, :] = np.nan

        for i in range(1, seq_len):
            # 维特比算法在每一时刻计算落到每一个隐状态的最大概率和路径
            # 并把他们暂存起来
            # 这里用到了矩阵化计算方法, 详见视频教程
            p_Obs_State = self.get_p_Obs_State(text[i])
            p_Obs_State = np.expand_dims(p_Obs_State, axis=0)
            prev_score = np.expand_dims(T1_table[i-1, :], axis=-1)
            # 广播算法, 发射概率和转移概率广播 + 转移概率
            curr_score = prev_score + self.transition + p_Obs_State
            # 存入T1 T2中
            T1_table[i, :] = np.max(curr_score, axis=0)
            T2_table[i, :] = np.argmax(curr_score, axis=0)
        # 回溯
        best_tag_id = int(np.argmax(T1_table[-1, :]))
        best_tags = [best_tag_id, ]
        for i in range(seq_len-1, 0, -1):
            best_tag_id = int(T2_table[i, best_tag_id])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

if __name__ == '__main__':
    model = HMM_NER(char2idx_path="./dicts/char2idx.json",
                    tag2idx_path="./dicts/tag2idx.json")
    model.fit("./corpus/train_data.txt")
    model.predict("我在中国吃美国的面包")