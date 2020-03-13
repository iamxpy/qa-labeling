from config.bert_config import cfg
from src import *
from torch.utils.data import DataLoader, RandomSampler
import random
import gc
import time
import pandas as pd
import numpy as np
import logging
import os

save_dir = "./ck_fold_5_epoch_5_LR_3e-05_layer_4_6"
print(cfg)


DATA_DIR = './google-quest-challenge'

sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
target_columns = sub.columns.values[1:].tolist()

NFOLDS = cfg["NFOLDS"]
BATCH_SIZE = cfg["BATCH_SIZE"]
swa_alpha = cfg["swa_alpha"]
# EPOCHS = cfg["EPOCHS"]
eval_every = cfg["EVAL_EVERY"]
LR = cfg["LR"]
MAX_SEQUENCE_LENGTH = cfg["MAX_SEQUENCE_LENGTH"]
Last_Layer = cfg["Last_Layer"]

if not os.path.exists(cfg["output"]):
    os.makedirs(cfg["output"])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 0
seed_everything(SEED)


def predict_result(model, test_loader):
    test_preds = np.zeros((len(test), len(target_columns)))

    test_len = len(test)
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, _ = data
            q_input_ids = q_input_ids.to(device)
            q_input_masks = q_input_masks.to(device)
            q_input_segments = q_input_segments.to(device)

            qa_input_ids = qa_input_ids.to(device)
            qa_input_masks = qa_input_masks.to(device)
            qa_input_segments = qa_input_segments.to(device)

            pred = model(q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks,
                         qa_input_segments)

            test_preds[idx * BATCH_SIZE: min((idx + 1) * BATCH_SIZE, test_len)] = pred.detach().cpu().numpy()

    test_preds_new = test_preds.copy()

    post_test_preds = deal_result(test_preds)
    other_test_preds = post_deal_result(test_preds_new)
    return test_preds, post_test_preds, other_test_preds


def get_layer_num(load_model_dir):
    for i in range(1, 30):
        tmp_str = "layer_" + str(i)
        if tmp_str in load_model_dir:
            return i
        tmp_str = "layer-" + str(i)
        if tmp_str in load_model_dir:
            return i
    return 0


def create_model(model_file, layer_num):
    model = BertForQuest()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(model_file))
    return model


def predict(test_loader):
    results = np.zeros((len(test), 30))
    other_results = np.zeros((len(test), 30))
    post_results = np.zeros((len(test), 30))

    swa_results = np.zeros((len(test), 30))
    swa_other_results = np.zeros((len(test), 30))
    swa_post_results = np.zeros((len(test), 30))

    files = os.listdir(save_dir)
    for file in files:
        model_file = os.path.join(save_dir, file)
        layer_num = get_layer_num(save_dir)

        print(model_file, layer_num)
        model = create_model(model_file, layer_num)

        if "swa" in model_file:
            swa_result, swa_post_result, swa_other_result = predict_result(model, test_loader)
            swa_results += swa_result
            swa_post_results += swa_post_result
            swa_other_results += swa_other_result
        else:
            result, post_result, other_result = predict_result(model, test_loader)
            results += result
            post_results += post_result
            other_results += other_result

    return results / NFOLDS, post_results / NFOLDS, other_results / NFOLDS, swa_results / NFOLDS, swa_post_results / NFOLDS, swa_other_results / NFOLDS


test = pd.read_csv(f'{DATA_DIR}/test.csv')
x_test = convert_lines(tokenizer, test['question_title'], test['question_body'], test['answer'])
test_loader = torch.utils.data.DataLoader(QuestDataset(x_test),
                                          batch_size=BATCH_SIZE, shuffle=False)

preds, post_preds, other_preds, swa_preds, swa_post_preds, swa_other_preds = predict(test_loader)


sub[target_columns] = preds
sub.to_csv("{}/result.csv".format(cfg["output"]), index=False)

sub[target_columns] = post_preds
sub.to_csv("{}/post.csv".format(cfg["output"]), index=False)

sub[target_columns] = other_preds
sub.to_csv("{}/other.csv".format(cfg["output"]), index=False)

sub[target_columns] = swa_preds
sub.to_csv("{}/swa_result.csv".format(cfg["output"]), index=False)

sub[target_columns] = swa_post_preds
sub.to_csv("{}/swa_post.csv".format(cfg["output"]), index=False)

sub[target_columns] = swa_other_preds
sub.to_csv("{}/swa_other.csv".format(cfg["output"]), index=False)
