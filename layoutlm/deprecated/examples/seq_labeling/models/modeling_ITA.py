# -*- coding: utf-8 -*-
import torch
from layoutlmft.models.layoutlmv2.modeling_layoutlmv2 import (
    LayoutLMv2PreTrainedModel,
    LayoutLMv2Config,
    LayoutLMv2Model
)
from torch import nn


class LayoutlmForImageTextMatching(LayoutLMv2PreTrainedModel):

    def __init__(self, config, max_seq_length):
        super(LayoutlmForImageTextMatching, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_seq_length = max_seq_length
        self.lmv2 = LayoutLMv2Model.from_pretrained(config.name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.decoder = nn.Linear(config.hidden_size, 1)
        self.proj = nn.Linear(max_seq_length + 49, config.num_labels)
        self.init_weights()

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
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            label=None,
    ):
        outputs = self.lmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])
        logits = self.decoder(sequence_output).squeeze(-1)

        # if attention_mask is not None:
        #     logits[:, :self.max_seq_length] = logits[:, :self.max_seq_length] * attention_mask

        active_logits = self.proj(logits)
        probs = logits.softmax(dim=1)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(probs.view(-1, self.num_labels), label.view(-1))
            return loss, probs
        else:
            return active_logits


class ResnetForImageTextMatching(nn.Module):

    def __init__(self, config: LayoutLMv2Config, max_seq_length):
        super(ResnetForImageTextMatching, self).__init__()
        self.config = config

        self.visual_encoder = torch.hub.load('/home/std2020/.cache/torch/hub/vision-0.10.0',
                                             'resnet18', pretrained=True,
                                             source='local',
                                             # num_classes=config.hidden_size,
                                             )

        self.lmv2 = LayoutLMv2Model.from_pretrained(config.name_or_path)
        self.visual_proj = nn.Linear(self.visual_encoder.layer4[1].conv2.out_channels, config.hidden_size)
        self.pixel_mean = torch.reshape(torch.tensor([103.5300, 116.2800, 123.6750]), (3, 1, 1))
        self.pixel_std = torch.tensor([[[57.3750]], [[57.1200]], [[58.3950]]])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, max_seq_length))
        self.decoder = nn.Linear(config.hidden_size, 1)
        self.proj = nn.Linear(max_seq_length, config.num_labels)

    def _cal_spatial_position_embeddings(self, bbox):
        try:
            left_position_embeddings = self.lmv2.embeddings.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.lmv2.embeddings.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.lmv2.embeddings.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.lmv2.embeddings.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.lmv2.embeddings.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.lmv2.embeddings.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def _cal_image_embedding(self, x):
        x = self.visual_encoder.conv1(x)
        x = self.visual_encoder.bn1(x)
        x = self.visual_encoder.relu(x)
        x = self.visual_encoder.maxpool(x)

        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)  # B*512*7*7

        x = torch.flatten(x, 2).transpose(1, 2)

        return self.visual_proj(x)

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = (
                torch.arange(
                    0,
                    1000 * (image_feature_pool_shape[1] + 1),
                    1000,
                    device=device,
                    dtype=bbox.dtype,
                )
                // self.config.image_feature_pool_shape[1]
        )
        visual_bbox_y = (
                torch.arange(
                    0,
                    1000 * (self.config.image_feature_pool_shape[0] + 1),
                    1000,
                    device=device,
                    dtype=bbox.dtype,
                )
                // self.config.image_feature_pool_shape[0]
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
        visual_bbox = self._calc_spatial_embeddings(visual_bbox)
        return visual_bbox

    def _calc_spatial_embeddings(self, bbox):
        try:
            left_position_embeddings = self.lmv2.embeddings.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.lmv2.embeddings.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.lmv2.embeddings.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.lmv2.embeddings.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.lmv2.embeddings.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.lmv2.embeddings.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def forward(self, input_ids, image, bbox, position_ids=None,
                attention_mask=None,
                label=None,
                token_type_ids=None,
                ):
        device = input_ids.device
        input_shape = input_ids.size()
        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)
        self.pixel_std = self.pixel_std.to(image.device)
        self.pixel_mean = self.pixel_mean.to(image.device)
        images = (image - self.pixel_mean) / self.pixel_std
        text_layout_emb = self.lmv2._calc_text_embeddings(input_ids, bbox, position_ids, token_type_ids)
        visual_box = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        img_emb = self._cal_image_embedding(images)
        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        pos_emb = self.lmv2.embeddings.position_embeddings(visual_position_ids)
        visual_emb = visual_box + img_emb + pos_emb
        visual_emb = self.avg_pool(visual_emb.transpose(1, 2)).transpose(1, 2)

        final_emb = text_layout_emb + visual_emb
        pooled_output = self.dropout(final_emb)
        logits = self.decoder(pooled_output).squeeze(-1)
        if attention_mask is not None:
            logits = logits * attention_mask

        active_logits = self.proj(logits)
        probs = active_logits.softmax(dim=1)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(probs.view(-1, self.config.num_labels), label.view(-1))
            return loss, probs
        else:
            return probs
