from transformers import *
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from config.bert_config import cfg
from src.common import *

class BertForQuest(BertPreTrainedModel):
    def __init__(self):
        super(BertForQuest, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config).from_pretrained(model_dir, config=config)
        self.bert_qa = BertModel(config).from_pretrained(model_dir, config=config)

        self.layer_num = cfg["Last_Layer"]

        self.head1 = nn.Sequential(
            nn.Linear(self.layer_num * cfg["hidden_size"], cfg["hidden_size"]),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(cfg["hidden_size"], 21),
        )

        self.head2 = nn.Sequential(
            nn.Linear(self.layer_num * cfg["hidden_size"], cfg["hidden_size"]),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(cfg["hidden_size"], 9),
        )

        print("*************qa9 output.***************")

        # self.init_weights()

    def forward(self, q_ids, q_attention_mask, q_seg_ids, qa_ids, qa_attention_mask, qa_seg_ids):
        _, _, q_hidden_states = self.bert(input_ids=q_ids, token_type_ids=q_seg_ids, attention_mask=q_attention_mask)
        q_h = []
        for i in range(1, self.layer_num + 1):
            q_h.append(q_hidden_states[-i][:, 0])

        q_h = torch.cat(q_h, 1)

        y1 = self.head1(q_h)

        _, _, qa_hidden_states = self.bert_qa(input_ids=qa_ids, token_type_ids=qa_seg_ids,
                                              attention_mask=qa_attention_mask)
        qa_h = []
        for i in range(1, self.layer_num + 1):
            qa_h.append(qa_hidden_states[-i][:, 0])

        qa_h = torch.cat(qa_h, 1)
        y2 = self.head2(qa_h)

        output = torch.cat([y1, y2], 1).sigmoid()
        return output


class PooledBertForQuest(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = bert_config.num_labels

        self.bert = BertModel(bert_config).from_pretrained(bertModel_dir, config=bert_config)
        self.bert_qa = BertModel(bert_config).from_pretrained(bertModel_dir, config=bert_config)

        self.layer_num = cfg["Last_Layer"]
        print("*************Pooled & Norm structure and qa30 output.***************")

        self.head1 = nn.Sequential(
            nn.Linear(768 * 8, 768 * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(768 * 4),
            nn.Dropout(0.2),
            nn.Linear(768 * 4, 21),
        )

        self.head2 = nn.Sequential(
            nn.Linear(768 * 8, 768 * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(768 * 4),
            nn.Dropout(0.2),
            nn.Linear(768 * 4, 30),
        )

        # self.init_weights()

    def forward(self, q_ids, q_attention_mask, q_seg_ids, qa_ids, qa_attention_mask, qa_seg_ids):
        outputs_q = self.bert(input_ids=q_ids, attention_mask=q_attention_mask, token_type_ids=q_seg_ids)

        outputs_qa = self.bert_qa(input_ids=qa_ids, attention_mask=qa_attention_mask, token_type_ids=qa_seg_ids)

        q_pooled_output_avg = torch.nn.functional.adaptive_avg_pool2d(outputs_q[0], (1, 768))
        q_pooled_output_avg = torch.squeeze(q_pooled_output_avg, 1)
        q_pooled_output_max = torch.nn.functional.adaptive_max_pool2d(outputs_q[0], (1, 768))
        q_pooled_output_max = torch.squeeze(q_pooled_output_max, 1)
        q_pooled_output = torch.cat([q_pooled_output_avg, q_pooled_output_max], 1)
        qa_pooled_output_avg = torch.nn.functional.adaptive_avg_pool2d(outputs_qa[0], (1, 768))
        qa_pooled_output_avg = torch.squeeze(qa_pooled_output_avg, 1)
        qa_pooled_output_max = torch.nn.functional.adaptive_max_pool2d(outputs_qa[0], (1, 768))
        qa_pooled_output_max = torch.squeeze(qa_pooled_output_max, 1)
        qa_pooled_output = torch.cat([qa_pooled_output_avg, qa_pooled_output_max], 1)
        q_mean_pool = torch.mean(torch.cat(outputs_q[2][-3:], 2), 1)
        q_max_pool, _ = torch.max(torch.cat(outputs_q[2][-3:], 2), 1)
        q_pooler_output = torch.cat([q_mean_pool, q_max_pool], 1)
        qa_mean_pool = torch.mean(torch.cat(outputs_qa[2][-3:], 2), 1)
        qa_max_pool, _ = torch.max(torch.cat(outputs_qa[2][-3:], 2), 1)
        qa_pooler_output = torch.cat([qa_mean_pool, qa_max_pool], 1)
        q_feature = torch.cat([q_pooled_output, q_pooler_output], 1)
        qa_feature = torch.cat([qa_pooled_output, qa_pooler_output], 1)
        y1 = self.head1(q_feature)
        y2 = self.head2(qa_feature)
        q_out = (y1 + y2[:, :21]) / 2
        output = torch.cat([q_out, y2[:, 21:]], 1).sigmoid()
        return output
