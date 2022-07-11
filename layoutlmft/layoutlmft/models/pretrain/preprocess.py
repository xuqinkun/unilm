import random
from pathlib import Path
import numpy as np
from PIL import Image

PROB_FOR_TIM = 0.2

PROB_FOR_COVER = 0.15

PROB_FOR_REPLACE = 0.1

PROB_OF_TOKEN_MASK = 0.8

PROB_FOR_MASK = 0.15

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


def replace_or_keep_image(match_image, image_shape):
    # [1123,794,3]
    h0 = image_shape[0]  # 1123
    w0 = image_shape[1]  # 794
    if match_image is not None:
        img = Image.open(match_image).convert("RGB")
        array = np.array(img)
        h = array.shape[0]  # 383
        w = array.shape[1]  # 717

        for l in range(len(box)):
            if flag[l] == 1 or cover[l] == 1:
                actual_b = actual_box[l]
                actual_x_min = int(actual_b[0] / h0 * h)
                actual_y_min = int(actual_b[1] / w0 * w)
                actual_x_max = int(actual_b[2] / h0 * h)
                actual_y_max = int(actual_b[3] / w0 * w)

                for i in range(actual_x_min, actual_x_max + 1):  # i的范围
                    for j in range(actual_y_min, actual_y_max + 1):  # j的范围
                        for k in range(0, 3):
                            if i < array.shape[1] and j < array.shape[0]:
                                array[j, i, k] = 0

        img = Image.fromarray(array, mode='RGB')
    else:
        zero_img = np.ones_like(image_shape) * 255
        img = Image.fromarray(zero_img, mode='RGB')
    return img


def _mask_tokens(tokens, mask):
    # 1.mask word
    # In MVLM, 15% text tokens are masked among which
    # 80% are replaced by a special token [MASK],
    # 10% are replaced by a random token sampled from the whole vocabulary,
    # and 10% remains the same.
    masked_tokens = []
    token_masks = []
    for token in tokens:
        prob_for_mask = random.random()
        temp_token = None
        if prob_for_mask < PROB_FOR_MASK:
            prob_for_mask /= PROB_FOR_MASK
            if prob_for_mask < PROB_OF_TOKEN_MASK:
                temp_token = masked_tokens
            elif prob_for_mask < PROB_OF_TOKEN_MASK + PROB_FOR_REPLACE:
                temp_token = str(random.randrange(250002))
        if temp_token is None:
            temp_token = token
        # 避免视觉线索泄漏，对图像中对应的区域进行了遮罩,在将原始页面图像输入到视觉编码器之前，将其屏蔽
        # 方便起见，flag=1表示此token为mask，否则不用
        if temp_token == mask:
            token_masks.append(1)
        else:
            token_masks.append(0)
        masked_tokens.append(temp_token)
    return masked_tokens, token_masks


def get_pretrain_data(data_dict, tokenizer):
    #######################
    return_token_list = []
    return_token_masks = []
    flags = []
    cover = []
    covers = []
    token = []
    tokens = []
    box = []
    boxes = []
    actual_box = []
    actual_boxes = []
    images = []
    match_images_label = []
    last_img = None
    pre_word = -1
    last_png = None
    for key, item in data_dict.items():
        tokens = item["tokens"]
        img_path = item['actual_path']
        labels = item['labels']
        bbox = item['bbox']
        actual_bbox = item['actual_bbox']
        masked_tokens, token_masks = _mask_tokens(tokens, tokenizer.mask_token)
        return_token_list.append(masked_tokens)
        return_token_masks.append(token_masks)

        # In TIA, 15% of the lines are covered.
        if word != pre_word:
            prob_of_tia = random.random()
            if prob_of_tia < PROB_FOR_COVER:
                prob_of_tia /= PROB_FOR_COVER
                cover.append(1)
            else:
                cover.append(0)
        pre_word = word
        if token:
            covers.append(cover)
            flags.append(flag)
            tokens.append(token)
            boxes.append(box)
            actual_boxes.append(actual_box)
            mask_tokens.append(mask_token)

            # 在图像中mask掉相应的flag=1的boxes区域后加入
            # [[1,2,3,4],[1,2,3,4],.....]
            img = Image.open(last_img).convert("RGB")
            print(img)
            array = np.array(img)

            # 1表示失配，0表示没有失配
            # In TIM, 15% images are replaced, and 5% are dropped.
            prob_of_tim = random.random()
            if prob_of_tim < 0.05:
                match_image = None
                match_images_label.append(1)
            elif prob_of_tim < 0.2:
                t = random.randrange(len(all_lines))
                match_image = all_lines[t]
            else:
                match_image = last_img
                match_images_label.append(0)

            img = replace_or_keep_image(match_image, img.shape)

            images.append(img)

            token = []
            flag = []
            box = []
            cover = []
            actual_box = []
            mask_token = []

            pre_word = -1

        for i in range(len(match_images_label)):
            if match_images_label[i] == 1:  # 表示失配
                for j in range(len(covers[i])):  # 全部标为[covered]
                    covers[i][j] = 1

    return tokens, mask_tokens, covers, match_images_label, boxes, actual_boxes, images
    # 分别用于   MVLM TIA TIM
