# coding=utf-8
from __future__ import absolute_import
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from scipy.stats import spearmanr
from modeling_bert import BertForSequenceClassification_linear, BertConfig
from bert_utils import BertTokenizer, compute_3sen_input_arrays


class Argument(object):
    pass


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
args = Argument()
args.model_name_or_path = "../input/linear-fold0/"
args.data_dir = os.path.join("../input/google-quest-challenge/")
args.per_gpu_eval_batch_size = 64
args.n_gpu = torch.cuda.device_count()
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
args.max_seq_length = 512
args.split_num = 1
args.lstm_hidden_size = 512
args.lstm_layers = 1
args.lstm_dropout = 0.1
args.dropout = 0.2
args.freeze = 0
args.do_train = False
args.do_eval = False
args.do_test = True
args.seed = 2019

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification_linear, BertTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

class InputExample(object):
    def __init__(self, q_title, q_body, answer, label=None):
        self.q_title = q_title
        self.q_body = q_body
        self.answer = answer
        self.label = label


class InputFeatures(object):
    def __init__(self, choices_features, label):
        self.choices_features = choices_features
        self.label = label


def read_examples(df: pd.DataFrame, feature_labels: list, label_columns: list = None):
    columns = list(feature_labels)
    if label_columns is not None:
        columns += label_columns
    examples = []
    for val in df[columns].values:
        if label_columns is not None:
            examples.append(InputExample(q_title=val[0], q_body=val[1], answer=val[2], label=val[3:]))
        else:
            examples.append(InputExample(q_title=val[0], q_body=val[1], answer=val[2]))
    return examples


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

# Set seed
set_seed(args)

df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
input_categories = list(df_train.columns[[1, 2, 5]])
output_categories = list(df_train.columns[11:])  # 30 categories
num_labels = 30

# Prepare model
tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

model = BertForSequenceClassification_linear.from_pretrained(args.model_name_or_path, args, config=config)
model.to(device)

if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare data
# Test data
test_ids, test_masks, test_segments = compute_3sen_input_arrays(df_test, input_categories, tokenizer,
                                                                args.max_seq_length)
predictions = []
test_ids_tensor = torch.tensor(test_ids, dtype=torch.long)
test_masks_tensor = torch.tensor(test_masks, dtype=torch.long)
test_segments_tensor = torch.tensor(test_segments, dtype=torch.long)

test_data = TensorDataset(test_ids_tensor, test_masks_tensor, test_segments_tensor)
# Run prediction for full data
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
model.eval()
for input_ids, input_mask, segment_ids in test_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        print("Predicting batch ......")
        batch_logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
    predictions.append(torch.sigmoid(batch_logits).cpu().numpy())
predictions = np.concatenate(predictions, 0)
df_predictions = pd.DataFrame(predictions, columns=output_categories)
df_test = pd.concat([df_test, df_predictions], axis=1)
df_test[['qa_id'] + output_categories].to_csv("submission.csv", index=False)
