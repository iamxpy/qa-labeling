from transformers import *
import torch
import pandas as pd
import numpy as np
from config.bert_config import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = cfg["model_dir"]

model_config = '{}/config.json'.format(model_dir)

pre_model = cfg["pretrained_model"]

vocab = '{}/vocab.txt'.format(model_dir)

if "albert" in pre_model:
    config = AlbertConfig.from_json_file(model_config)
    tokenizer = AlbertTokenizer.from_pretrained(model_dir, do_lower_case=False)
elif "roberta" in pre_model:
    config = RobertaConfig.from_json_file(model_config)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, do_lower_case=True)
elif "bert" in pre_model:
    config = BertConfig.from_json_file(model_config)
    if "uncased" in pre_model:
        tokenizer = BertTokenizer.from_pretrained(vocab, do_lower_case=True)
    else:
        print("cased")
        tokenizer = BertTokenizer.from_pretrained(vocab, do_lower_case=False)
else:
    raise ValueError(f"Not supported model: {pre_model}")

config.num_labels = 30
config.output_hidden_states = True

all_result = []
train_data = pd.read_csv('./google-quest-challenge/folds_trans_more.csv')

for j in range(30):
    tem = []
    for i in range(len(train_data)):
        if not train_data.loc[i][11 + j] in tem:
            tem.append(train_data.loc[i][11 + j])
    tem.sort()
    all_result.append(tem)

all_result = np.array(all_result)


def find(a, result):
    min = 100
    t = 0
    for i in range(len(result)):
        gap = abs(result[i] - a)
        if gap < min:
            min = gap
            t = result[i]
    return t


def deal_result(all_predictions, ban_list=None):
    all_predictions_new = []
    idx_max = [0 for _ in range(30)]

    for i in range(len(all_predictions)):
        tem = []
        for j in range(30):
            if ban_list is not None and j in ban_list:
                tem.append(all_predictions[i][j])
            else:
                tem.append(find(all_predictions[i][j], all_result[j]))
                if all_predictions[i][j] > all_predictions[idx_max[j]][j]:
                    idx_max[j] = i
        tem = np.array(tem)
        all_predictions_new.append(tem)

    all_predictions_new = np.array(all_predictions_new)
    for j in range(30):
        if len(np.unique(all_predictions_new[:, j])) == 1:
            all_predictions_new[idx_max[j]][j] += 1e-7

    return all_predictions_new


all_result_v2 = []
fenmu = [6, 9, 10, 15]
for f in fenmu:
    for i in range(f):
        a = i / f
        if not a in all_result_v2:
            all_result_v2.append(a)
all_result_v2.append(1.0)


def post_deal_result(all_predictions, ban_list=None):
    all_predictions_new = []
    idx_max = [0 for _ in range(30)]
    for i in range(len(all_predictions)):
        tem = []
        for j in range(30):
            if ban_list is not None and j in ban_list:
                tem.append(all_predictions[i][j])
            else:
                if all_predictions[i][j] > all_predictions[idx_max[j]][j]:
                    idx_max[j] = i
                tem.append(find(all_predictions[i][j], all_result_v2))

        tem = np.array(tem)
        all_predictions_new.append(tem)

    all_predictions_new = np.array(all_predictions_new)
    for j in range(30):
        if len(np.unique(all_predictions_new[:, j])) == 1:
            all_predictions_new[idx_max[j]][j] += 1e-7

    return all_predictions_new


# labels.npy保存了每列的各标签（例如spelling那列只有0,1/3,2/3三个标签）的频率，利用这些数据求出每列标签的分布prior_probs_list
classes = np.load('./google-quest-challenge/labels.npy', allow_pickle=True)
prior_freqs_list = [np.array([classes[i][key] for key in sorted(classes[i])]) for i in range(len(classes))]
prior_probs_list = [freqs / sum(freqs) for freqs in prior_freqs_list]


def deal_column(s: np.ndarray, freq):
    """
    align假设模型能预测出样本标签的相对大小（而不必接近真实值），这样我们就可以结合训练集中每列标签的分布来对齐该列
    另外一个假设是训练集和测试集的标签分布相差不大（从目前的结果来看的确是的）
    s给出原值，freq按照标签值从小到大的顺序给出各个标签的频次，因为比赛的指标只与值的排名有关，
    所以可以将同属一个标签的样本的值对齐到这个标签的最小值。例子：如果某列在训练集只有3种标签，0, 1/3, 2/3，且分布为[0.5, 0.2, 0.3]
    假设对该列的原预测s为[0.01,0.03,0.05,0.02,0.07,0.04,0.09,0.0,0.08,0.06],由于0标签在该样本集中理论上有5个（10*0.5=5）
    将最低的5个映射为该标签的最小值0，对于1/3和2/3的处理也类似，因此处理之后的结果为：
    [0.0,0.0,0.05,0.0,0.07,0.0,0.07,0.0,0.07,0.05]
    """
    res = s.copy()  # use a copy to return
    d = {i: v for i, v in enumerate(s)}  # <下标,原值>
    d = sorted(d.items(), key=lambda item: item[1])
    j = 0
    for i in range(len(freq)):
        if freq[i] > 0 and j < len(d):
            fixed_value = d[j][1]
            while freq[i] > 0:
                res[d[j][0]] = fixed_value
                freq[i] -= 1
                j += 1
    return res


# prob是训练集中的标签分布，n是当前需要处理的数据集（验证集或测试集）样本数量
def estimate_frequency(prob: np.ndarray, n):
    tmp = prob * n  # 直接将概率乘以样本数然后round并不能确保求和仍为n，所以少了的几个需要合理分配到各标签
    # 此处做法是考虑round时的误差，例如1.9取整为2误差为0.1,而1.5取整为2误差就是0.5
    freq = [int(round(t)) for t in tmp]
    confidence = {i: np.abs(0.5 - (x - int(x))) for i, x in enumerate(tmp)}  # 小数点第一位距离5越远，取整的"confidence"越大
    confidence = sorted(confidence.items(), key=lambda item: item[1])
    # fix frequency according to confidence of 'round' operation
    fix_order = [idx for idx, _ in confidence]
    idx = 0
    s = np.sum(freq)
    # 修复各类的样本数，直到总和等于所需样本数n
    while s != n:
        if s > n:
            freq[fix_order[idx]] -= 1
        else:
            freq[fix_order[idx]] += 1
        s = np.sum(freq)
        # 理论上最多每个类都加/减一个就可以了，因为round的误差不可能超过n，不过这里写的是循环修改
        idx = (idx + 1) % len(fix_order)
    # 如果结果是只有一个标签的样本数不为0（其实不可能出现，最极端的spelling列都不可能，不过考虑这个鲁棒点）
    # 修改为有两个标签含有非零个样本，一个有n-1个样本，另一个只有1个样本。
    if np.sum(np.array(freq) > 0) < 2:  # in case there is only one class
        freq[0], freq[len(freq) - 1] = n - 1, 1
    return freq


def align(predictions, ban_list=None):
    num_samples = predictions.shape[0]  # 得到当前需要后处理的样本数量
    predictions_new = predictions.copy()
    for i in range(30):
        # 处理每一列，跳过会降低分数的列
        if ban_list is not None and i in ban_list:
            continue
        frequency = estimate_frequency(prior_probs_list[i], num_samples)
        predictions_new[:, i] = deal_column(predictions[:, i], frequency)
    return predictions_new
