import numpy as np
import torch
from torch.utils import data
from math import floor, ceil
from config.bert_config import cfg

MAX_LEN = 512

QA_TITLE_MAX_LEN = 30
QA_BODY_MAX_LEN = 239
QA_ANSWER_MAX_LEN = 239

Q_TITLE_MAX_LEN = 50
Q_BODY_MAX_LEN = 458


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def qa_get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def q_get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """RoBERTa does not make use of token type ids, therefore a list of zeros is returned."""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [0] * max_seq_length


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def qa_trim_input(tokenizer, title, body, answer, max_seq_length=512):
    all_title = []
    all_body = []
    all_answer = []
    for t, b, a in zip(title, body, answer):

        tokenizer_t = tokenizer.tokenize(t, add_prefix_space=True)
        tokenizer_b = tokenizer.tokenize(b, add_prefix_space=True)
        tokenizer_a = tokenizer.tokenize(a, add_prefix_space=True)

        t_len = len(tokenizer_t)
        b_len = len(tokenizer_b)
        a_len = len(tokenizer_a)

        t_max_len = QA_TITLE_MAX_LEN
        b_max_len = QA_BODY_MAX_LEN
        a_max_len = QA_ANSWER_MAX_LEN

        if (t_len + b_len + a_len + 4) > max_seq_length:

            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
                b_max_len = b_max_len + ceil((t_max_len - t_len) / 2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len
                b_new_len = b_max_len + (a_max_len - a_len)
            elif b_max_len > b_len:
                a_new_len = a_max_len + (b_max_len - b_len)
                b_new_len = b_len
            else:
                a_new_len = a_max_len
                b_new_len = b_max_len

            if t_new_len + a_new_len + b_new_len + 4 != max_seq_length:
                raise ValueError("New sequence length should be %d, but is %d" % (
                    max_seq_length, (t_new_len + a_new_len + b_new_len + 4)))

            if cfg["choose"] == "head":
                tokenizer_t = tokenizer_t[:t_new_len]
                tokenizer_b = tokenizer_b[:b_new_len]
                tokenizer_a = tokenizer_a[:a_new_len]
            elif cfg["choose"] == "tail":
                tokenizer_t = tokenizer_t[-t_new_len:]
                tokenizer_b = tokenizer_b[-b_new_len:]
                tokenizer_a = tokenizer_a[-a_new_len:]
            elif cfg["choose"] == "both":
                t_head_len = int(t_new_len * cfg["ratio"])
                t_tail_len = t_new_len - t_head_len

                b_head_len = int(b_new_len * cfg["ratio"])
                b_tail_len = b_new_len - b_head_len

                a_head_len = int(a_new_len * cfg["ratio"])
                a_tail_len = a_new_len - a_head_len

                tokenizer_t = tokenizer_t[:t_head_len] + tokenizer_t[-t_tail_len:]
                tokenizer_b = tokenizer_b[:b_head_len] + tokenizer_b[-b_tail_len:]
                tokenizer_a = tokenizer_a[:a_head_len] + tokenizer_a[-a_tail_len:]

                assert t_new_len == t_head_len + t_tail_len
                assert t_new_len == t_head_len + t_tail_len
                assert a_new_len == a_head_len + a_tail_len

        all_title.append(tokenizer_t)
        all_body.append(tokenizer_b)
        all_answer.append(tokenizer_a)

    return all_title, all_body, all_answer


def q_trim_input(tokenizer, title, body, max_seq_length=512):
    all_title = []
    all_body = []
    for t, b in zip(title, body):

        tokenizer_t = tokenizer.tokenize(t, add_prefix_space=True)
        tokenizer_b = tokenizer.tokenize(b, add_prefix_space=True)

        t_len = len(tokenizer_t)
        b_len = len(tokenizer_b)

        t_max_len = Q_TITLE_MAX_LEN
        b_max_len = Q_BODY_MAX_LEN

        if (t_len + b_len + 4) > max_seq_length:

            if t_max_len > t_len:
                t_new_len = t_len
                b_max_len = max_seq_length - 4 - t_new_len
            else:
                t_new_len = t_max_len

            if b_max_len > b_len:
                b_new_len = b_len
            else:
                b_new_len = b_max_len

            if t_new_len + b_new_len + 4 != max_seq_length:
                raise ValueError("New sequence length should be %d, but is %d" % (
                    max_seq_length, (t_new_len + b_new_len + 4)))

            if cfg["choose"] == "head":
                tokenizer_t = tokenizer_t[:t_new_len]
                tokenizer_b = tokenizer_b[:b_new_len]
            elif cfg["choose"] == "tail":
                tokenizer_t = tokenizer_t[-t_new_len:]
                tokenizer_b = tokenizer_b[-b_new_len:]
            elif cfg["choose"] == "both":
                t_head_len = int(t_new_len * cfg["ratio"])
                t_tail_len = t_new_len - t_head_len

                b_head_len = int(b_new_len * cfg["ratio"])
                b_tail_len = b_new_len - b_head_len

                tokenizer_t = tokenizer_t[:t_head_len] + tokenizer_t[-t_tail_len:]
                tokenizer_b = tokenizer_b[:b_head_len] + tokenizer_b[-b_tail_len:]

                assert t_new_len == t_head_len + t_tail_len
                assert t_new_len == t_head_len + t_tail_len

        all_title.append(tokenizer_t)
        all_body.append(tokenizer_b)

    return all_title, all_body


def convert_lines(tokenizer, title, body, answer, max_seq_length=512):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    q_title, q_body = q_trim_input(tokenizer, title, body)

    qa_title, qa_body, qa_answer = qa_trim_input(tokenizer, title, body, answer)

    q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments = [], [], [], [], [], []
    for i, (q_t, q_b, t, b, a) in enumerate(zip(q_title, q_body, qa_title, qa_body, qa_answer)):
        q_stoken = [CLS] + q_t + [SEP, SEP] + q_b + [SEP]

        q_ids = get_ids(q_stoken, tokenizer, max_seq_length)
        q_masks = get_masks(q_stoken, max_seq_length)
        q_segments = get_segments(q_stoken, max_seq_length)
        q_input_ids.append(q_ids)
        q_input_masks.append(q_masks)
        q_input_segments.append(q_segments)

        qa_stoken = [CLS] + t + b + [SEP, SEP] + a + [SEP]
        qa_ids = get_ids(qa_stoken, tokenizer, max_seq_length)
        qa_masks = get_masks(qa_stoken, max_seq_length)
        qa_segments = get_segments(qa_stoken, max_seq_length)
        qa_input_ids.append(qa_ids)
        qa_input_masks.append(qa_masks)
        qa_input_segments.append(qa_segments)

    return [
        torch.from_numpy(np.asarray(q_input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(q_input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(q_input_segments, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(qa_input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(qa_input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(qa_input_segments, dtype=np.int32)).long(),
    ]


class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, targets=None):
        self.q_input_ids = x_train[0]
        self.q_input_masks = x_train[1]
        self.q_input_segments = x_train[2]

        self.qa_input_ids = x_train[3]
        self.qa_input_masks = x_train[4]
        self.qa_input_segments = x_train[5]

        self.targets = targets if targets is not None else np.zeros((x_train[0].shape[0], 30))

    def __getitem__(self, idx):
        q_input_ids = self.q_input_ids[idx]
        q_input_masks = self.q_input_masks[idx]
        q_input_segments = self.q_input_segments[idx]

        qa_input_ids = self.qa_input_ids[idx]
        qa_input_masks = self.qa_input_masks[idx]
        qa_input_segments = self.qa_input_segments[idx]

        target = self.targets[idx]
        return q_input_ids, q_input_masks, q_input_segments, qa_input_ids, qa_input_masks, qa_input_segments, target

    def __len__(self):
        return len(self.q_input_ids)
