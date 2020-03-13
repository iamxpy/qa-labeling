import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
import gc
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
import os
import random
import time
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import operator
from nltk.tokenize.treebank import TreebankWordTokenizer
import spacy
from spacy.lang.en import English
from scipy.stats import spearmanr
import re
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

tqdm.pandas()

# 三种不同的提取词干的方法
ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer('english')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

NFOLDS = 5
CRAWL_EMBEDDING_PATH = './crawl-300d-2M.pkl'
GLOVE_EMBEDDING_PATH = './glove.840B.300d.pkl'
train_csv_path = './google-quest-challenge/folds_trans_more.csv'
test_csv_path = './google-quest-challenge/test.csv'
seed = 0
epochs = 100
max_features = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(seed)
# 将文章切分成句子
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

train = pd.read_csv(train_csv_path)
test = pd.read_csv(test_csv_path)

# 宾夕法尼亚州立大学 Treebank单词分割器
tree_tokenizer = TreebankWordTokenizer()


def handle_contractions(x):
    x = tree_tokenizer.tokenize(x)
    x = ' '.join(x)
    return x


for col in ['question_body', 'question_title', 'answer']:
    train[col] = train[col].apply(lambda x: handle_contractions(x))
    test[col] = test[col].apply(lambda x: handle_contractions(x))

tokenizer = text.Tokenizer(lower=True)

X_train_question = train['question_body']
X_train_title = train['question_title']
X_train_answer = train['answer']

X_test_question = test['question_body']
X_test_title = test['question_title']
X_test_answer = test['answer']

tokenizer.fit_on_texts(list(X_train_question) + \
                       list(X_train_answer) + \
                       list(X_train_title) + \
                       list(X_test_question) + \
                       list(X_test_answer) + \
                       list(X_test_title))


def split_document(texts):
    all_sents = []
    for text in tqdm(texts):
        doc = nlp(text)
        sents = []
        for idx, sent in enumerate(doc.sents):
            sents.append(sent.text)
        all_sents.append(sents)

    return all_sents


X_train_question = split_document(X_train_question)
X_train_answer = split_document(X_train_answer)

X_test_question = split_document(X_test_question)
X_test_answer = split_document(X_test_answer)


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, 'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix(word_index, path):
    embeddings_index = load_embeddings(path)

    embedding_matrix = np.zeros((max_features + 1, 300))
    unknown_words = []

    for key, i in word_index.items():
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        unknown_words.append(word)

    return embedding_matrix, unknown_words


# word_index: 字典，将单词（字符串）映射为它们的排名或者索引。
crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
print('n unknown words (crawl): ', len(unknown_words_crawl))

glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('n unknown words (glove): ', len(unknown_words_glove))

embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
print(embedding_matrix.shape)

del crawl_matrix
del glove_matrix
gc.collect()


def add_question_metadata_features(text):
    doc = nlp(text)
    indirect = 0
    question_count = 0
    reason_explanation_words = 0
    choice_words = 0

    for sent in doc.sents:
        if '?' in sent.text and '?' == sent.text[-1]:
            question_count += 1  # -> question_multi_intent
            for token in sent:
                if token.text.lower() == 'why':  # question_type_reason_explanation e.g index->102
                    reason_explanation_words += 1
                elif token.text.lower() == 'or':
                    choice_words += 1  # question_type_choice
    if question_count == 0:
        indirect = 1

    return [indirect, question_count, reason_explanation_words, choice_words]


ans_user_category = train[train[['answer_user_name', 'category']].duplicated()][
    ['answer_user_name', 'category']].values.tolist()
print(len(ans_user_category))


def question_answer_author_same(df):
    q_username = df['question_user_name']
    a_username = df['answer_user_name']

    author_same = []
    for i in range(len(df)):
        if q_username[i] == a_username[i]:
            author_same.append(int(1))
        else:
            author_same.append(int(0))

    return author_same


def add_external_features(df):
    # If the question is longer, it may be more clear, which may help users give a more
    df['question_body'] = df['question_body'].progress_apply(lambda x: str(x))
    df['question_num_words'] = df.question_body.str.count('\S+')

    # The assumption here is that longer answer could bring more useful detail
    df['answer'] = df['answer'].progress_apply(lambda x: str(x))
    df['answer_num_words'] = df.answer.str.count('\S+')

    # if the question is long and the answer is short, it may be less relevant
    df["question_vs_answer_length"] = df['question_num_words'] / df['answer_num_words']

    # if answer's author is the same as the corresponding question's author,
    # Why he/she asked question.. :)
    df["q_a_author_same"] = question_answer_author_same(df)

    # answers which was posted by users who answer one category more than one times, they may have read more similar questions.
    # thus, the answers by this type of user will more relevent to question.
    ans_user_cat = []
    for x in tqdm(df[['answer_user_name', 'category']].values.tolist()):
        if x in ans_user_category:
            ans_user_cat.append(int(1))
        else:
            ans_user_cat.append(int(0))
    df['ans_user_with_cat'] = ans_user_cat

    handmade_features = []

    for idx, text in enumerate(df['question_body'].values):
        handmade_features.append(add_question_metadata_features(text))

    return df, np.array(handmade_features)


train, train_handmade_features = add_external_features(train)
test, test_handmade_features = add_external_features(test)

num_words_scaler = MinMaxScaler()
num_words_scaler.fit(train[['question_num_words', 'answer_num_words']].values)
train[['question_num_words', 'answer_num_words']] = num_words_scaler.transform(
    train[['question_num_words', 'answer_num_words']].values)
test[['question_num_words', 'answer_num_words']] = num_words_scaler.transform(
    test[['question_num_words', 'answer_num_words']].values)

train_external_features = train[['question_num_words', 'answer_num_words',
                                 "question_vs_answer_length", "q_a_author_same",
                                 "ans_user_with_cat"]].values
test_external_features = test[['question_num_words', 'answer_num_words',
                               "question_vs_answer_length", "q_a_author_same",
                               "ans_user_with_cat"]].values

train_external_features = np.hstack((train_external_features, train_handmade_features))
test_external_features = np.hstack((test_external_features, test_handmade_features))


def tokenizer_to_index(texts, max_number_sentence, maxlen):
    all_seqs = []

    for text in tqdm(texts):
        seqs = []
        for sent in text:
            sent = tokenizer.texts_to_sequences(pd.Series(sent))
            sent = pad_sequences(sent, maxlen=maxlen)
            if len(sent) == 0:
                seqs.append([0] * maxlen)
            else:
                seqs.append(sent[0])
        if len(seqs) < max_number_sentence:
            gap = max_number_sentence - len(seqs)
            pad_zeros = [[0] * maxlen for g in range(gap)]
            seqs = pad_zeros + seqs  # pad -> pre
        elif len(seqs) > max_number_sentence:
            seqs = seqs[:max_number_sentence]

        all_seqs.append(np.array(seqs))
    return np.stack(all_seqs, 0)


X_train_question = tokenizer_to_index(X_train_question, max_number_sentence=20, maxlen=50)
X_train_answer = tokenizer_to_index(X_train_answer, max_number_sentence=20, maxlen=50)

X_test_question = tokenizer_to_index(X_test_question, max_number_sentence=20, maxlen=50)
X_test_answer = tokenizer_to_index(X_test_answer, max_number_sentence=20, maxlen=50)

X_train_title = tokenizer.texts_to_sequences(X_train_title)
X_train_title = pad_sequences(X_train_title, maxlen=30)

X_test_title = tokenizer.texts_to_sequences(X_test_title)
X_test_title = pad_sequences(X_test_title, maxlen=30)

# the assumption here is that the question comment relevance might depend on the category of the question

unique_categories = list(set(train['category'].unique().tolist() + test['category'].unique().tolist()))
category_dict = {i + 1: e for i, e in enumerate(unique_categories)}
category_dict_reverse = {v: k for k, v in category_dict.items()}

unique_hosts = list(set(train['host'].unique().tolist() + test['host'].unique().tolist()))
host_dict = {i + 1: e for i, e in enumerate(unique_hosts)}
host_dict_reverse = {v: k for k, v in host_dict.items()}

train_host = train['host'].apply(lambda x: host_dict_reverse[x]).values
train_category = train['category'].apply(lambda x: category_dict_reverse[x]).values

test_host = test['host'].apply(lambda x: host_dict_reverse[x]).values
test_category = test['category'].apply(lambda x: category_dict_reverse[x]).values

n_cat = len(category_dict) + 1
cat_emb = min(np.ceil((len(category_dict)) / 2), 50)
n_host = len(host_dict) + 1
host_emb = min(np.ceil((len(host_dict)) / 2), 50)


class QuestDataset(Dataset):

    def __init__(self, df, questions, answers, titles, hosts, categories, external_features):
        self.df = df
        self.questions = questions
        self.answers = answers
        self.titles = titles
        self.hosts = hosts
        self.categories = categories
        self.external_features = external_features

        self.question_cols = ['question_asker_intent_understanding',
                              'question_body_critical', 'question_conversational',
                              'question_expect_short_answer', 'question_fact_seeking',
                              'question_has_commonly_accepted_answer',
                              'question_interestingness_others', 'question_interestingness_self',
                              'question_multi_intent', 'question_not_really_a_question',
                              'question_opinion_seeking', 'question_type_choice',
                              'question_type_compare', 'question_type_consequence',
                              'question_type_definition', 'question_type_entity',
                              'question_type_instructions', 'question_type_procedure',
                              'question_type_reason_explanation', 'question_type_spelling',
                              'question_well_written']
        self.answer_cols = ['answer_helpful', 'answer_level_of_information',
                            'answer_plausible', 'answer_relevance',
                            'answer_satisfaction', 'answer_type_instructions',
                            'answer_type_procedure', 'answer_type_reason_explanation',
                            'answer_well_written']

        self.label = self.df[self.question_cols + self.answer_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        title = self.titles[idx]
        host = self.hosts[idx]
        category = self.categories[idx]
        external_features = self.external_features[idx]

        labels = self.label[idx]

        return [question, answer, title, host, category, external_features], labels


class QuestDataset_test(Dataset):

    def __init__(self, questions, answers, titles, hosts, categories, external_features):
        self.questions = questions
        self.answers = answers
        self.titles = titles
        self.hosts = hosts
        self.categories = categories
        self.external_features = external_features

    def __len__(self):
        return self.questions.shape[0]

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        title = self.titles[idx]
        host = self.hosts[idx]
        category = self.categories[idx]
        external_features = self.external_features[idx]

        return [question, answer, title, host, category, external_features]


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1)


class rnn_Layer(nn.Module):

    def __init__(self, input_dim, output_dim, max_len):
        super().__init__()
        self.lstm_1 = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)
        self.atten = Attention(output_dim * 2, max_len)

    def forward(self, x):
        lstm_output, _ = self.lstm_1(x)

        return self.atten(lstm_output)


class QuestModel(nn.Module):

    def __init__(self, embedding_matrix, n_cat, cat_emb, n_host, host_emb):
        super().__init__()

        LSTM_UNITS = 128
        embed_size = embedding_matrix.shape[1]
        DENSE_HIDDEN_UNITS = LSTM_UNITS * 4
        # max_features = config.MAX_FEATURES

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.category_embedding = nn.Embedding(n_cat, int(cat_emb))
        self.host_embedding = nn.Embedding(n_host, int(host_emb))

        ##########################################################
        # LSTM
        ##########################################################
        self.lstm_q_1 = rnn_Layer(embed_size, LSTM_UNITS, max_len=50)
        self.lstm_q_2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm_a_1 = rnn_Layer(embed_size, LSTM_UNITS, max_len=50)
        self.lstm_a_2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm_t_1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.p_fc1 = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
                                   nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5))
        self.a_fc1 = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
                                   nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5))
        self.t_fc1 = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS),
                                   nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(0.5))

        ######################################
        # Q-branch
        ######################################
        self.q_t_consine = nn.CosineSimilarity(dim=1)
        self.q_fc1 = nn.Sequential(
            nn.Linear(DENSE_HIDDEN_UNITS * 2 + int(cat_emb) + int(host_emb) + 6, DENSE_HIDDEN_UNITS),
            nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))
        self.q_fc2 = nn.Linear(DENSE_HIDDEN_UNITS, 21)

        ######################################
        # QA-branch
        ######################################

        self.aq_bil = nn.Bilinear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.aq_fc1 = nn.Sequential(nn.Linear(DENSE_HIDDEN_UNITS * 4 + 4, DENSE_HIDDEN_UNITS),
                                    nn.BatchNorm1d(DENSE_HIDDEN_UNITS),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5))

        self.aq_fc2 = nn.Linear(DENSE_HIDDEN_UNITS, 9)

    def forward(self, question, answer, title, host, category, external_features):

        _, q_sentence_num, q_max_len = question.size()
        _, a_sentence_num, a_max_len = answer.size()
        category_embed = self.category_embedding(category)
        host_embed = self.host_embedding(host)
        question_length = external_features[:, 0].unsqueeze(-1)
        answer_length = external_features[:, 1].unsqueeze(-1)
        q_vs_a = external_features[:, 2].unsqueeze(-1)
        qa_same_author = external_features[:, 3].unsqueeze(-1)
        a_with_cat = external_features[:, 4].unsqueeze(-1)

        indirect = external_features[:, 5].unsqueeze(-1)
        num_question = external_features[:, 6].unsqueeze(-1)
        reasonal_explain = external_features[:, 7].unsqueeze(-1)
        choice = external_features[:, 8].unsqueeze(-1)

        #######################################
        # Question
        #######################################
        q_reps = []
        for i in range(q_sentence_num):
            question_sentence = question[:, i, :].long()  # (batch_size, max_len)
            question_embedding = self.embedding(question_sentence)
            question_embedding = self.embedding_dropout(question_embedding)  # (batch_size, max_len, embed_size)
            q_sentence_reps = self.lstm_q_1(question_embedding)  # (batch_size, output_dim*2) #Word-level-attention
            q_sentence_reps = torch.unsqueeze(q_sentence_reps, dim=1)  # (batch_size, 1, LSTM_UNITS*2)
            q_reps.append(q_sentence_reps)

        q_reps = torch.cat(q_reps, dim=1)  # (batch_size, sentence_num, LSTM_UNITS*2)
        q_lstm2, _ = self.lstm_q_2(q_reps)

        q_avg_pool = torch.mean(q_lstm2, 1)
        q_max_pool, _ = torch.max(q_lstm2, 1)

        #######################################
        # answer
        #######################################
        a_reps = []
        for j in range(a_sentence_num):
            answer_sentence = answer[:, j, :].long()  # (batch_size, max_len)
            answer_embedding = self.embedding(answer_sentence)
            answer_embedding = self.embedding_dropout(answer_embedding)  # (batch_size, max_len, embed_size)
            a_sentence_reps = self.lstm_a_1(answer_embedding)  # (batch_size, LSTM_UNITS*2)
            a_sentence_reps = torch.unsqueeze(a_sentence_reps, dim=1)  # (batch_size, 1, DENSE_HIDDEN_UNITS)
            a_reps.append(a_sentence_reps)

        a_reps = torch.cat(a_reps, dim=1)  # (batch_size, sentence_num, DENSE_HIDDEN_UNITS)
        a_lstm2, _ = self.lstm_a_2(a_reps)

        a_avg_pool = torch.mean(a_lstm2, 1)
        a_max_pool, _ = torch.max(a_lstm2, 1)

        #######################################
        # title
        #######################################

        title_embedding = self.embedding(title.long())
        title_embedding = self.embedding_dropout(title_embedding)

        t_lstm1, _ = self.lstm_t_1(title_embedding)

        t_avg_pool = torch.mean(t_lstm1, 1)
        t_max_pool, _ = torch.max(t_lstm1, 1)

        q_features = self.p_fc1(
            torch.cat((q_max_pool, q_avg_pool), 1))  # (batch_size, LSTM_UNITS*4) -> (batch_size, LSTM_UNITS)
        a_features = self.a_fc1(
            torch.cat((a_max_pool, a_avg_pool), 1))  # (batch_size, LSTM_UNITS*4) -> (batch_size, LSTM_UNITS)
        t_features = self.t_fc1(
            torch.cat((t_max_pool, t_avg_pool), 1))  # (batch_size, LSTM_UNITS*4) -> (batch_size, LSTM_UNITS)
        ######################################
        # Q-branch
        ######################################
        cosine_q_t = self.q_t_consine(q_features, t_features).unsqueeze(-1)
        hidden_q = self.q_fc1(torch.cat((q_features, t_features, category_embed, host_embed,
                                         cosine_q_t, question_length, indirect, num_question,
                                         reasonal_explain, choice), 1))
        q_result = self.q_fc2(hidden_q)
        ######################################
        # QA-branch
        ######################################
        bil_sim = self.aq_bil(q_features, a_features)
        hidden_aq = self.aq_fc1(torch.cat((q_features, t_features, a_features,
                                           bil_sim, answer_length, q_vs_a,
                                           qa_same_author, a_with_cat), 1))
        aq_result = self.aq_fc2(hidden_aq)

        return torch.cat((q_result, aq_result), 1)


def train_model(train_loader, optimizer, criterion):
    model.train()
    avg_loss = 0.

    for idx, (inputs, labels) in enumerate(train_loader):
        questions, answers, titles, hosts, categories, external_features = inputs
        questions, answers, titles, hosts, categories = questions.cuda(), answers.cuda(), titles.cuda(), hosts.long().cuda(), categories.long().cuda()
        external_features = external_features.float().cuda()
        labels = labels.float().cuda()

        optimizer.zero_grad()
        output_train = model(questions, answers, titles, hosts, categories, external_features)
        loss = criterion(output_train, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)

    return avg_loss


def val_model(val_loader):
    avg_val_loss = 0.
    model.eval()
    preds = []
    original = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            questions, answers, titles, hosts, categories, external_features = inputs
            questions, answers, titles, hosts, categories = questions.cuda(), answers.cuda(), titles.cuda(), hosts.long().cuda(), categories.long().cuda()
            external_features = external_features.float().cuda()
            labels = labels.float().cuda()

            output_val = model(questions, answers, titles, hosts, categories, external_features)
            avg_val_loss += criterion(output_val, labels).item() / len(val_loader)
            preds.append(output_val.cpu().numpy())
            original.append(labels.cpu().numpy())

        score = 0
        for i in range(30):
            score += np.nan_to_num(
                spearmanr(np.concatenate(original)[:, i], np.concatenate(preds)[:, i]).correlation / 30)

    return avg_val_loss, score


def predict_result(model, test_loader, batch_size=64):
    output = np.zeros((len(test_set), 30))
    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            start_index = idx * batch_size
            end_index = min(start_index + batch_size, len(test_set))
            questions, answers, titles, hosts, categories, external_features = inputs
            questions, answers, titles, hosts, categories = questions.cuda(), answers.cuda(), titles.cuda(), hosts.long().cuda(), categories.long().cuda()
            external_features = external_features.float().cuda()
            predictions = model(questions, answers, titles, hosts, categories, external_features)
            predictions = torch.sigmoid(predictions)
            output[start_index:end_index, :] = predictions.detach().cpu().numpy()

    return output


kf = KFold(n_splits=5, shuffle=True, random_state=seed)
test_set = QuestDataset_test(X_test_question,
                             X_test_answer,
                             X_test_title,
                             test_host,
                             test_category,
                             test_external_features,
                             )
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
result = np.zeros((len(test), 30))

for cv in range(NFOLDS):
    print(f'fold {cv + 1}')
    train_df = train[train['fold'] != cv]
    val_df = train[train['fold'] == cv]
    train_index = train[train['fold'] != cv].index
    val_index = train[train['fold'] == cv].index

    train_set = QuestDataset(train_df,
                             X_train_question[train_index],
                             X_train_answer[train_index],
                             X_train_title[train_index],
                             train_host[train_index],
                             train_category[train_index],
                             train_external_features[train_index])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    val_set = QuestDataset(val_df,
                           X_train_question[val_index],
                           X_train_answer[val_index],
                           X_train_title[val_index],
                           train_host[val_index],
                           train_category[val_index],
                           train_external_features[val_index])
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = QuestModel(embedding_matrix, n_cat, cat_emb, n_host, host_emb)
    model.to(device)

    best_avg_loss = 100.0
    best_score = 0.0
    best_param_loss = None
    best_param_score = None
    i = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    for epoch in range(epochs):

        if i == 5: break
        start_time = time.time()
        avg_loss = train_model(train_loader, optimizer, criterion)
        avg_val_loss, score = val_model(val_loader)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(epoch + 1, epochs,
                                                                                                     avg_loss,
                                                                                                     avg_val_loss,
                                                                                                     score,
                                                                                                     elapsed_time))

        if best_avg_loss > avg_val_loss:
            i = 0
            best_avg_loss = avg_val_loss
            best_param_loss = model.state_dict()
        if best_score < score:
            best_score = score
            best_param_score = model.state_dict()
        else:
            i += 1
        scheduler.step(avg_val_loss)

    model.load_state_dict(best_param_score)
    result += predict_result(model, test_loader)

    torch.cuda.empty_cache()
    del train_df
    del val_df
    del model
    gc.collect()

result /= 5

