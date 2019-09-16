import json

def load_dict(path):
    """
    加载字典
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_data(path):
    """
    读取txt文件, 加载训练数据
    :param path:
    :return:
    [{'text': ['当', '希', '望', ...],
     'label': ... }, {...}, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [eval(i) for i in f.readlines()]