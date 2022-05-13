import os

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from layoutlmft.models.layoutlmv2.modeling_layoutlmv2 import \
    (LayoutLMv2Model, LayoutLMv2PreTrainedModel)
import torch


class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.vocab_size  # 改为词表大小
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear1 = nn.Linear(config.hidden_size, self.num_labels)
        self.linear2 = nn.Linear(config.hidden_size, 2)
        self.linear3 = nn.Linear(config.hidden_size, 2)
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.post_init()

        # 是否冻结参数
        for param in self.layoutlmv2.parameters():
            param.requires_grad = False

    def save_prompt(self):
        return self.prefix_encoder.state_dict()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.layoutlmv2.device)
        ###########################

        past_key_values = self.prefix_encoder(prefix_tokens)  # [4,32,2*12*768]
        # print(batch_size)   4
        # print(self.n_layer) 12
        # print(self.n_head)  12
        # print(self.n_embd)  64  12*64=768
        past_key_values = past_key_values.view(batch_size, self.pre_seq_len, self.n_layer * 2, self.n_head, self.n_embd)
        # [4,32,24,12,64]
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # [24,4,12,32,64] -> [2,4,12,32,64]
        return past_key_values

    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    def forward(
            self,
            input_ids=None,
            bbox=None,
            image=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            ##########
            tokens=None,
            TIA_MASK=None,
            covers=None,
            match_label=None,
            mvlm_mask=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]

        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.layoutlmv2.device)
        attention_mask = torch.cat((attention_mask, prefix_attention_mask), dim=1)

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # only take the text part of the output representations

        sequence_output = outputs[0][:, :seq_length]

        # torch.Size([2, 512, 768])
        sequence_output = self.dropout(sequence_output)

        logits = self.linear1(sequence_output)

        Alogits = self.linear2(sequence_output)

        Mlogits = self.linear3(sequence_output[:, 0])

        # torch.Size([2, 512, 250002])
        loss_fct = CrossEntropyLoss()
        mvlm_mask = (mvlm_mask.view(-1) == 1)
        active_logits = logits.view(-1, self.num_labels)[mvlm_mask]
        active_labels = tokens.view(-1)[mvlm_mask]
        loss_mvlm = loss_fct(active_logits, active_labels)
        # <s>和</s>不用预测
        active_loss = (TIA_MASK.view(-1) == 1)  # 舍弃掉MVLM中MASK部分
        Aactive_logits = Alogits.view(-1, 2)[active_loss]
        Aactive_labels = covers.view(-1)[active_loss]
        loss_tia = loss_fct(Aactive_logits, Aactive_labels)
        Mactive_logits = Mlogits.view(-1, 2)
        Mactive_labels = match_label.view(-1)
        loss_tim = loss_fct(Mactive_logits, Mactive_labels)
        loss = loss_mvlm + loss_tia + loss_tim

        return (TokenClassifierOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, ), self.save_prompt())
