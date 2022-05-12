import random

import numpy as np
from PIL import Image

'''
MVLM部分仅利用token与layout(bbox)信息
TIA会利用visu信息，但是对于MVLM的MASK部分忽略不计
TIM由于不同图片纵横比不一致，因此替换要bbox坐标时可能会超出图片范围
A   [MASK]  [COVERED](TIA loss不算)      [MATCH]
B   [B]     [COVERED]                   [MATCH]  
C   [100]   [COVERED]                   [MATCH]
D   [D]     [COVERED]                   [MATCH]

I   [MASK]  [NOT COVERED](TIA loss不算)   [MATCH]
你  [B]     [NOT COVERED]                 [MATCH]
E   [E]     [NOT COVERED]                 [MATCH]

N   [N]     [COVERED]       [NOT MATCH]
T   [T]     [COVERED]       [NOT MATCH]
张   [C]     [COVERED]       [NOT MATCH]
'''

data_dir='/mnt/disk2/nlp_data/invoice/data/train_sep_pretrain.txt'

with open(data_dir,'r', encoding='utf-8') as f:
    total_images=[]
    #######################先获取完整的image列表
    for line in f:
        if line.startswith("-DOCSTART-") or line =="" or line =="\n":
            if last_img:
                # 在图像中mask掉相应的flag=1的boxes区域后加入
                #[[1,2,3,4],[1,2,3,4],.....]
                last_img = last_img.rstrip()
                total_images.append(last_img)
        else:
            splits=line.split('\t')
            last_img=splits[5].replace("\n", "")
def get_pretrain_data():
    with open(data_dir, 'r', encoding='utf-8') as f:
        #######################
        mask_token=[]
        mask_tokens=[]
        flag=[]
        flags=[]
        cover=[]
        covers=[]
        token=[]
        tokens=[]
        box=[]
        boxes=[]
        actual_box=[]
        actual_boxes=[]
        images=[]
        match_images_label=[]
        last_img=None
        tip=0

        pre_word=-1

        for line in f:
            if line.startswith("-DOCSTART-") or line =="" or line =="\n":
                if token:

                    covers.append(cover)
                    flags.append(flag)
                    tokens.append(token)
                    boxes.append(box)
                    actual_boxes.append(actual_box)
                    mask_tokens.append(mask_token)

                    # 在图像中mask掉相应的flag=1的boxes区域后加入
                    #[[1,2,3,4],[1,2,3,4],.....]
                    last_img = last_img.rstrip()
                    img = Image.open(last_img).convert("RGB")
                    array = np.array(img)
                    #[1123,794,3]
                    h0=array.shape[0]#1123
                    w0=array.shape[1]#794

                    match_image=None
                    #################################1表示失配，0表示没有失配
                    prob3 = random.random()
                    if prob3 < 0.2:
                        prob3 /= 0.2

                        if prob1 < 0.75:
                            t=random.randrange(len(total_images))
                            match_image=total_images[t]
                        else:
                            match_image =None
                        match_images_label.append(1)
                    else:
                        match_image=last_img
                        match_images_label.append(0)

                    if match_image is not None:
                        img = Image.open(match_image).convert("RGB")
                        array = np.array(img)
                        h = array.shape[0]  # 383
                        w = array.shape[1]  # 717

                        for l in range(len(box)):
                            if flag[l]==1 or cover[l]==1:
                                actual_b=actual_box[l]
                                actual_x_min=int(actual_b[0]/h0*h)
                                actual_y_min=int(actual_b[1]/w0*w)
                                actual_x_max=int(actual_b[2]/h0*h)
                                actual_y_max=int(actual_b[3]/w0*w)

                                for i in range(actual_x_min,actual_x_max+1):#i的范围
                                    for j in range(actual_y_min,actual_y_max+1):#j的范围
                                        for k in range(0,3):
                                            if i<array.shape[1] and j<array.shape[0]:
                                                array[j,i,k]=0

                        img = Image.fromarray(array, mode='RGB')
                    else:
                        zero_img=np.ones_like(array)*255
                        img=Image.fromarray(zero_img,mode='RGB')

                    images.append(img)

                    token=[]
                    flag=[]
                    box=[]
                    cover=[]
                    actual_box=[]
                    mask_token=[]

                    pre_word=-1
            else:
                splits=line.split('\t')
                token.append(splits[0])
                box.append([int(b) for b in splits[2].split()])
                actual_box.append([int(b) for b in splits[3].split()])
                last_img=splits[5].replace("\n", "")

                #######################################MVLM
                #1.mask word
                prob1 = random.random()
                if prob1 < 0.15:
                    prob1 /= 0.15

                    if prob1 < 0.8:
                        temp = '<mask>'

                    elif prob1 < 0.9:
                        temp = str(random.randrange(250002))

                    else:
                        temp= splits[0]
                else:
                    temp = splits[0]
                mask_token.append(temp)
                #2. 避免视觉线索泄漏，对图像中对应的区域进行了遮罩,在将原始页面图像输入到视觉编码器之前，将其屏蔽
                #方便起见，flag=1表示此token为mask，否则不用
                if temp=='<mask>':

                    flag.append(1)
                else:
                    flag.append(0)
                ######################################TIA

                word=splits[-1].replace("\n","")

                if word!=pre_word:
                    prob2 = random.random()
                    if prob2<0.15:
                        prob2 /= 0.15
                        tip=1
                    else:
                        tip=0
                pre_word=word
                cover.append(tip)

    for i in range(len(match_images_label)):
        if match_images_label[i]==1:#表示失配
            for j in range(len(covers[i])):#全部标为[covered]
                covers[i][j]=1

    return tokens,mask_tokens,covers,match_images_label,boxes,actual_boxes,images
    #分别用于   MVLM TIA TIM





