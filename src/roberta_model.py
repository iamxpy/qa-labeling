import torch.nn as nn
from src.common import *


class RobertaForQuest(BertPreTrainedModel):
    def __init__(self):
        super(RobertaForQuest, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = RobertaModel.from_pretrained(model_dir, config=config)
        self.bert_qa = RobertaModel.from_pretrained(model_dir, config=config)

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
