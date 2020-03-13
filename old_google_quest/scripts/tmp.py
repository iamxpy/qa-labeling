class Mymodel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.bert = BertModel(bert_config).from_pretrained(model_path, config=bert_config)
        self.bert_qa = BertModel(bert_config).from_pretrained(model_path, config=bert_config)
        self.head1 = nn.Sequential(
            # nn.LayerNorm(self.bert.config.hidden_size * 9),
            # nn.Dropout(0.2),

            nn.Linear(self.bert.config.hidden_size * 8, self.bert_qa.config.hidden_size * 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.bert.config.hidden_size * 4),
            # nn.BatchNorm1d(self.bert.config.hidden_size * 4, eps=1e-05, momentum=0.1, affine=True,
            #                track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(self.bert.config.hidden_size * 4, 21),
        )
        # print(t_num)
        self.head2 = nn.Sequential(
            nn.Linear(self.bert_qa.config.hidden_size * 8, self.bert_qa.config.hidden_size * 4),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(self.bert.config.hidden_size * 4, eps=1e-05, momentum=0.1, affine=True,
            #                track_running_stats=True),
            nn.LayerNorm(self.bert_qa.config.hidden_size * 4),
            nn.Dropout(0.2),
            nn.Linear(self.bert_qa.config.hidden_size * 4, 30),
        )
        # self.init_weights()

    def forward(self, q_ids, q_seg_id, qa_ids, qa_seg_id, attention_mask=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        q_attention_mask = (q_ids > 0)
        qa_attention_mask = (qa_ids > 0)
        # a_attention_mask = (a_ids>0)

        outputs_q = self.bert(q_ids,
                              attention_mask=q_attention_mask,
                              token_type_ids=q_seg_id,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        outputs_qa = self.bert_qa(qa_ids,
                                  attention_mask=qa_attention_mask,
                                  token_type_ids=qa_seg_id,
                                  position_ids=position_ids,
                                  head_mask=head_mask,
                                  inputs_embeds=inputs_embeds)

        q_pooled_output_avg = torch.nn.functional.adaptive_avg_pool2d(outputs_q[0], (1, self.bert.config.hidden_size))
        q_pooled_output_avg = torch.squeeze(q_pooled_output_avg, 1)
        q_pooled_output_max = torch.nn.functional.adaptive_max_pool2d(outputs_q[0], (1, self.bert.config.hidden_size))
        q_pooled_output_max = torch.squeeze(q_pooled_output_max, 1)
        q_pooled_output = torch.cat([q_pooled_output_avg, q_pooled_output_max], 1)
        qa_pooled_output_avg = torch.nn.functional.adaptive_avg_pool2d(outputs_qa[0],
                                                                       (1, self.bert_qa.config.hidden_size))
        qa_pooled_output_avg = torch.squeeze(qa_pooled_output_avg, 1)
        qa_pooled_output_max = torch.nn.functional.adaptive_max_pool2d(outputs_qa[0],
                                                                       (1, self.bert_qa.config.hidden_size))
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