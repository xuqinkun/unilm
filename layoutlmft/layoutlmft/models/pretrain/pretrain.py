import os
import os
import random

import numpy as np
import torch
from transformers import LayoutXLMProcessor, AutoConfig, AdamW, \
    get_linear_schedule_with_warmup

from LMmodel import LayoutLMv2ForTokenClassification
from preprocess import get_pretrain_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_save_dir='/home/hejiabang/soft_prompt/'
data_pre_dir='/mnt/disk2/nlp_data/invoice/pre_data'

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)

tokens,mask_tokens,covers,match_images_label,boxes,actual_boxes,images=get_pretrain_data()

config = AutoConfig.from_pretrained(
        'microsoft/layoutxlm-base',
        revision="main",
    )

config.hidden_dropout_prob = 0.2
config.pre_seq_len = 126
config.prefix_projection = False
config.prefix_hidden_size = 512  ################# 512
config.has_relative_attention_bias=True
config.has_spatial_attention_bias=True

model = LayoutLMv2ForTokenClassification.from_pretrained(
            'microsoft/layoutxlm-base',
            config=config,
            revision="main", )

model.to(device)


def convert_examples_to_features(
       tokens, words, bboxes, covers,max_seq_len=512,
):
    final_words=[]
    final_boxes=[]
    final_tokens=[]
    final_covers=[]

    for token , word, bbox , cover in zip(tokens, words, bboxes,covers):

        padding_length = (max_seq_len-2) - len(word)#padding 到 510

        word += ["<pad>"] * padding_length
        token += ["<pad>"] * padding_length

        cover+=[0] * padding_length

        bbox += [[0, 0, 0, 0]] * padding_length

        final_tokens.append(token)
        final_words.append(word)
        final_boxes.append(bbox)
        final_covers.append(cover)


    return final_tokens,final_words,final_boxes,final_covers

train_tokens,train_words,train_boxes,covers=convert_examples_to_features(tokens,mask_tokens,boxes,covers)


processor=LayoutXLMProcessor.from_pretrained('microsoft/layoutxlm-base',
                                             apply_ocr=False)


def Process(images,words,boxes,tokens,covers):

    encoded_inputs={}
    encoded_input=processor(text=words,word_labels=covers,boxes=boxes,images=images,
                             padding="max_length",truncation=True,max_length=512)
    encoded_input2=processor(text=tokens,word_labels=covers,boxes=boxes,images=images,
                              padding="max_length",truncation=True,max_length=512)

    encoded_inputs['image'] = torch.tensor(encoded_input['image'], device=device)

    #################################包含mask
    #250001
    #16（情况1）
    #250001  250001
    # 6     16      6   3024（请况2）
    #从头往后两个一起遍历，列表1和列表2，列表1为mask后的分词输出(指针a)，列表2为原始分词输出(指针b)
    #a和b同时往后走，
    #################################
    mask_tokens=encoded_input['input_ids']#用token的指针

    tokens=encoded_input2['input_ids']
    final_mask_tokens=np.copy(encoded_input2['input_ids'])#深拷贝

    mvlm_mask=np.zeros_like(tokens)
    tia_mask=np.ones_like(tokens)#cover
    ##################################################
    for batch in range(len(tokens)):
        mask_token=mask_tokens[batch]
        token=tokens[batch]
        i=0#mask token
        j=0#token
        while i<len(mask_token) and j<len(token):
            if mask_token[i]==250001 and token[j]!=6:
                final_mask_tokens[batch][j]=250001
                mvlm_mask[batch][j]=1
                tia_mask[batch][j]=0
            elif mask_token[i]==250001 and token[j]==6:
                tia_mask[batch][j]=0
                mvlm_mask[batch][j] = 1
                j+=1
                final_mask_tokens[batch][j] = 250001
                tia_mask[batch][j]=0
                mvlm_mask[batch][j] = 1
            i+=1
            j+=1

    #################################
    encoded_inputs['input_ids'] = torch.tensor(final_mask_tokens, device=device)
    encoded_inputs['attention_mask'] = torch.tensor(encoded_input2['attention_mask'], device=device)
    encoded_inputs['bbox'] = torch.tensor(encoded_input2['bbox'], device=device)
    encoded_inputs['tokens']=torch.tensor(tokens, device=device)
    encoded_inputs['labels']=torch.tensor(encoded_input2['labels'],device=device)
    encoded_inputs['mvlm_mask']=torch.tensor(mvlm_mask, device=device)
    encoded_inputs['tia_mask']=torch.tensor(tia_mask,device=device)

    return encoded_inputs

from torch.utils.data import Dataset,DataLoader

class V2Dataset(Dataset):
    def __init__(self,encoded_inputs):
        self.all_images=encoded_inputs['image']
        self.all_input_ids=encoded_inputs['input_ids']
        self.all_attention_masks=encoded_inputs['attention_mask']
        self.all_bboxes=encoded_inputs['bbox']
        self.all_tokens=encoded_inputs['tokens']
        self.all_mvlm_mask=encoded_inputs['mvlm_mask']
        self.all_tia_mask=encoded_inputs['tia_mask']
        self.all_labels=encoded_inputs['labels']
        self.all_match_labels=torch.tensor(match_images_label,device=device)

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, index):
        return (
            self.all_images[index],
            self.all_input_ids[index],
            self.all_attention_masks[index],
            self.all_bboxes[index],
            self.all_tokens[index],
            self.all_tia_mask[index],
            self.all_labels[index],
            self.all_match_labels[index],
            self.all_mvlm_mask[index]
        )

train_dataset = V2Dataset(Process(images, train_words, train_boxes, train_tokens,covers))
#2个数据
train_dataloader=DataLoader(train_dataset,batch_size=1)

num_train_epochs = 200
max_grad_norm=1.0


optimizer=AdamW(model.parameters(),lr=3e-3)
num_train_steps=num_train_epochs*len(train_dataloader)
wm_steps=num_train_steps*0.01#warm_up
scheduler=get_linear_schedule_with_warmup(optimizer, num_warmup_steps=wm_steps, num_training_steps=num_train_steps)

for epoch in range(num_train_epochs):
    model.train()
    print("Epoch:",epoch)

    tr_loss = 0.0
    global_step=0
    for idx,batch in enumerate(train_dataloader):

        batch_size=batch[1].shape[0]

        temp=torch.zeros_like(batch[8]).cuda()#mvlm==0表示该处没有mask的地方，直接忽略
        if batch[8].equal(temp):
            continue

        outputs=model.forward(
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

        loss=outputs[0].loss
        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm(
                parameters=model.parameters(),
                max_norm=max_grad_norm
            )

        tr_loss+=loss.item()

        optimizer.step()
        scheduler.step()

        global_step+=1

        if global_step%50==0:
            print(f"re_loss mean :{tr_loss / global_step}")

    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)

    torch.save(outputs[1], output_save_dir+'prompt_.pkl')  # save entire net


