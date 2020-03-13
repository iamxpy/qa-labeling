from transformers import *
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from config.bert_config import cfg
from src.common import *

class ALBertForAnswer(nn.Module):
    def __init__(self):
        super().__init__()
        config.output_hidden_states = True

        self.bert = AlbertModel(config).from_pretrained(model_dir, config=config)

        self.layer_num = cfg["Last_Layer"]

        self.head = nn.Sequential(
            nn.Linear(self.layer_num * cfg["hidden_size"], 768),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(768, 9),
        )

        print("*************qa9 output.***************")

    def forward(self, qa_ids, qa_attention_mask, qa_seg_ids):

        _, _, qa_hidden_states = self.bert(input_ids=qa_ids, token_type_ids=qa_seg_ids,
                                              attention_mask=qa_attention_mask)
        qa_h = []
        for i in range(1, self.layer_num + 1):
            qa_h.append(qa_hidden_states[-i][:, 0])

        qa_h = torch.cat(qa_h, 1)
        output = self.head(qa_h).sigmoid()

        return output


class ALBertForQuest(nn.Module):
    def __init__(self):
        super().__init__()
        config.output_hidden_states = True

        self.bert = AlbertModel(config).from_pretrained(model_dir, config=config)

        self.layer_num = cfg["Last_Layer"]

        self.head = nn.Sequential(
            nn.Linear(self.layer_num * cfg["hidden_size"], 768),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(768, 21),
        )

        print("*************qa21 output.***************")

    def forward(self, q_ids, q_attention_mask, q_seg_ids):
        _, _, q_hidden_states = self.bert(input_ids=q_ids, token_type_ids=q_seg_ids, attention_mask=q_attention_mask)

        q_h = []
        for i in range(1, self.layer_num + 1):
            q_h.append(q_hidden_states[-i][:, 0])

        q_h = torch.cat(q_h, 1)

        output = self.head(q_h).sigmoid()

        return output
