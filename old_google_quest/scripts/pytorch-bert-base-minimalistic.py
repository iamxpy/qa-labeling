# coding: utf-8

# get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')
# get_ipython().system('pip install ../input/transformers/transformers-master/ > /dev/null')


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from tqdm import trange
from torch.utils import data
from transformers import (
    BertTokenizer, BertModel
)
from transformers import AdamW
from transformers import BertConfig
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


train = pd.read_csv("../input/google-quest-challenge/train.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")

target_cols = ['question_asker_intent_understanding', 'question_body_critical',
               'question_conversational', 'question_expect_short_answer',
               'question_fact_seeking', 'question_has_commonly_accepted_answer',
               'question_interestingness_others', 'question_interestingness_self',
               'question_multi_intent', 'question_not_really_a_question',
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity',
               'question_type_instructions', 'question_type_procedure',
               'question_type_reason_explanation', 'question_type_spelling',
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible',
               'answer_relevance', 'answer_satisfaction',
               'answer_type_instructions', 'answer_type_procedure',
               'answer_type_reason_explanation', 'answer_well_written']

# From the Ref Kernel's
from math import floor, ceil


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
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


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=120, a_max_len=358):
    # 293+239+30 = 508 + 4 = 512
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" % (
                max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    print("Preprocessing .....")
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        np.asarray(input_ids, dtype=np.int64),
        np.asarray(input_masks, dtype=np.int64),
        np.asarray(input_segments, dtype=np.int64)
    ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


tokenizer = BertTokenizer.from_pretrained('../input/bert-base-uncased/vocab.txt')
input_categories = list(train.columns[[1, 2, 5]])

outputs = compute_output_arrays(train, columns=target_cols)
inputs = compute_input_arays(train, input_categories, tokenizer, max_sequence_length=512)
test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)

lengths = np.argmax(inputs[0] == 0, axis=1)
lengths[lengths == 0] = inputs[0].shape[1]
y_train_torch = torch.tensor(train[target_cols].values, dtype=torch.float32)

lengths = torch.tensor(lengths).numpy()
y_train_torch = y_train_torch.numpy()

X_tr_inputs_ids, X_val_inputs_ids, X_tr_masks, X_val_masks, X_tr_inputs_segs, X_val_inputs_segs, y_train, y_val, X_tr_lengths, X_val_lengths = train_test_split(
    inputs[0], inputs[1], inputs[2], y_train_torch, lengths,
    test_size=0.30, random_state=46)

X_tr_inputs_ids = torch.tensor(X_tr_inputs_ids)
X_val_inputs_ids = torch.tensor(X_val_inputs_ids)

X_tr_masks = torch.tensor(X_tr_masks, dtype=torch.long)
X_val_masks = torch.tensor(X_val_masks, dtype=torch.long)

X_tr_inputs_segs = torch.tensor(X_tr_inputs_segs)
X_val_inputs_segs = torch.tensor(X_val_inputs_segs)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_tr_lengths = torch.tensor(X_tr_lengths)
X_val_lengths = torch.tensor(X_val_lengths)

# Select a batch size for training
batch_size = 2

train_data = TensorDataset(X_tr_inputs_ids, X_tr_masks, X_tr_inputs_segs, y_train, X_tr_lengths)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(X_val_inputs_ids, X_val_masks, X_val_inputs_segs, y_val, X_val_lengths)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

next(iter(train_dataloader))  # input_ids, input_masks, input_segments, targets, lengths


class BertSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(BertSequenceClassification, self).__init__()
        self.num_labels = num_labels
        bert_model_config = '../input/bert-base-uncased/config.json'
        bert_config = BertConfig.from_json_file(bert_model_config)
        bert_config.num_labels = 30
        bert_bin_file_path = '../input/bert-base-uncased/'
        self.bert = BertModel.from_pretrained(bert_bin_file_path, config=bert_config)
        self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(768, num_labels)
        self.loss_fct = BCEWithLogitsLoss()

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        mean_last_hidden_state = self.dropout(mean_last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def pool_hidden_state(self, last_hidden_state):
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state


model = BertSequenceClassification(num_labels=len(y_train[0]))


def train(model, num_epochs, optimizer, train_dataloader, valid_dataloader, train_loss_set=[], valid_loss_set=[],
          lowest_eval_loss=None, start_epoch=0, device="cpu"
          ):
    """
    Train the model and save the model with the lowest validation loss
    """
    crit_function = nn.BCEWithLogitsLoss()
    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        t = tqdm(total=len(train_data), desc="Training: ", position=0)
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_input_segs, b_labels, b_lengths = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_input_segs, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            t.update(n=b_input_ids.shape[0])
        t.close()
        # Update tracking variables
        epoch_train_loss = tr_loss / num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables 
        eval_loss = 0
        num_eval_samples = 0

        v_preds = []
        v_labels = []

        # Evaluate data for one epoch
        t = tqdm(total=len(validation_data), desc="Validating: ", position=0)
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_input_segs, b_labels, b_lengths = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                preds = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_input_segs)
                loss = crit_function(preds, b_labels)
                v_labels.append(b_labels.cpu().numpy())
                v_preds.append(preds.cpu().numpy())
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)
            t.update(n=b_labels.shape[0])
        t.close()

        v_labels = np.vstack(v_labels)
        v_preds = np.vstack(v_preds)
        print(v_labels.shape)
        print(v_preds.shape)
        rho_val = np.mean([spearmanr(v_labels[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0]),
                                     v_preds[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0])).correlation for ind
                           in range(v_preds.shape[1])]
                          )
        epoch_eval_loss = eval_loss / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Epoch #{}, training BCE loss: {}, validation BCE loss: ~{}, validation spearmanr: {}".format(0,
                                                                                                            epoch_train_loss,
                                                                                                            epoch_eval_loss,
                                                                                                            rho_val))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
        #   save_model(model, model_save_path, actual_epoch,\
        #              lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
            # save model
            # save_model(model, model_save_path, actual_epoch,\
            #            lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")

    return model, train_loss_set, valid_loss_set


optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1, correct_bias=False)
model, train_loss_set, valid_loss_set = train(model=model, num_epochs=7,
                                              optimizer=optimizer,
                                              train_dataloader=train_dataloader,
                                              valid_dataloader=validation_dataloader,
                                              device='cuda' if torch.cuda.is_available() else 'cpu'
                                              )

sequences = np.array(test_inputs[0])
lengths = np.argmax(sequences == 0, axis=1)
lengths[lengths == 0] = sequences.shape[1]

dataset = data.TensorDataset(torch.tensor(test_inputs[0]),
                             torch.tensor(test_inputs[1], dtype=torch.long),
                             torch.tensor(test_inputs[2]),
                             )

test_dataloader = data.DataLoader(dataset,
                                  batch_size=8,
                                  shuffle=False,
                                  drop_last=False
                                  )

next(iter(test_dataloader))


def generate_predictions(model, dataloader, num_labels, device="cpu", batch_size=8):
    pred_probs = np.array([]).reshape(0, num_labels)

    model.to(device)
    model.eval()

    for X, masks, segments in dataloader:
        X = X.to(device)
        masks = masks.to(device)
        segments = segments.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks, token_type_ids=segments)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])
    return pred_probs


pred_probs = generate_predictions(model, test_dataloader, num_labels=30, device="cuda", batch_size=8)

df_submit = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
df_submit[target_cols] = pred_probs

df_submit.to_csv("submission.csv", index=False)
