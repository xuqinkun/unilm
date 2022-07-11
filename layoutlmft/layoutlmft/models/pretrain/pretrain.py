import os
import os
import random
from pathlib import Path
from layoutlmft.models.pretrain.LMmodel import LayoutLMv2ForTokenClassification
from layoutlmft.models.pretrain.dataset import V2Dataset
from layoutlmft.models.pretrain.preprocess import get_pretrain_data
from layoutlmft.models.pretrain.utils import *
from torch.utils.data import DataLoader
from transformers import LayoutXLMProcessor, AutoConfig, AdamW, \
    get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_save_dir = '/home/std2020/xuqinkun/data/invoice/soft_prompt/'
data_pre_dir = '/home/std2020/xuqinkun/data/invoice'
data_root_dir = Path('/home/std2020/xuqinkun/data/invoice')
data_dir = '/home/std2020/xuqinkun/data/invoice/data/train_sep.txt'


def train():
    for epoch in range(num_train_epochs):
        model.train()
        print("Epoch:", epoch)
        tr_loss = 0.0
        global_step = 0
        for idx, batch in enumerate(train_dataloader):

            batch_size = batch[1].shape[0]
            # mvlm==0表示该处没有mask的地方，直接忽略
            temp = torch.zeros_like(batch[8]).cuda()
            if batch[8].equal(temp):
                continue

            outputs = model.forward(
                input_ids=batch[1],
                bbox=batch[3],
                image=batch[0],
                attention_mask=batch[2],
                ########################
                tokens=batch[4],
                TIA_MASK=batch[5],
                covers=batch[6],
                match_label=batch[7],
                mvlm_mask=batch[8]
            )

            loss = outputs[0].loss
            optimizer.zero_grad()
            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm(
                    parameters=model.parameters(),
                    max_norm=max_grad_norm
                )

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % 50 == 0:
                print(f"re_loss mean :{tr_loss / global_step}")

        if not os.path.exists(output_save_dir):
            os.makedirs(output_save_dir)

        torch.save(outputs[1], output_save_dir + 'prompt_.pkl')  # save entire net


if __name__ == '__main__':
    torch.manual_seed(123456)
    torch.cuda.manual_seed(123456)
    np.random.seed(123456)
    random.seed(123456)

    processor = LayoutXLMProcessor.from_pretrained('microsoft/layoutxlm-base',
                                                   apply_ocr=False)
    data_dict = read_lines(data_dir, data_root_dir)
    tokens, mask_tokens, covers, match_images_label, boxes, actual_boxes, images = get_pretrain_data(data_dict, processor.tokenizer)
    train_tokens, train_words, train_boxes, covers = convert_examples_to_features(tokens, mask_tokens, boxes, covers)

    config = AutoConfig.from_pretrained(
        'microsoft/layoutxlm-base',
        revision="main",
    )

    config.hidden_dropout_prob = 0.2
    config.pre_seq_len = 126
    config.prefix_projection = False
    config.prefix_hidden_size = 512
    config.has_relative_attention_bias = True
    config.has_spatial_attention_bias = True

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        'microsoft/layoutxlm-base',
        config=config,
        revision="main", )

    model.to(device)
    train_dataset = V2Dataset(Process(processor, device, images, train_words, train_boxes, train_tokens, covers), device)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    num_train_epochs = 200
    max_grad_norm = 1.0

    optimizer = AdamW(model.parameters(), lr=3e-3)
    num_train_steps = num_train_epochs * len(train_dataloader)
    wm_steps = num_train_steps * 0.01  # warm_up
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=wm_steps,
                                                num_training_steps=num_train_steps)
