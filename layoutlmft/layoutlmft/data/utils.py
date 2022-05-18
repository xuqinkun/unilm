import cv2
import numpy as np
import torch
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList


def normalize_bbox(bbox, size):
    bbox = simplify_bbox(bbox)
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)


def drawPolygon(ImShape, Polygon, Color):
    Im = np.zeros(ImShape, np.uint8)
    try:
        cv2.fillPoly(Im, Polygon, Color)
    except:
        try:
            cv2.fillConvexPoly(Im, Polygon, Color)
        except:
            print('canot fill')
    return Im


def get2PolyInterSectAreaSize(ImShape, Polygon1, Polygon2):
    Im1 = drawPolygon(ImShape, np.array(Polygon1), 122)
    Im2 = drawPolygon(ImShape, np.array(Polygon2), 133)
    Im = Im1 + Im2
    ret, OverlapIm = cv2.threshold(Im, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(OverlapIm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret_size = (0, 0)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        ret_size = (w, h)
    return ret_size

class Word:

    def __init__(self, line_id=None, x_pos=0, y_pos=0, word='', need=True):
        self._line_id = line_id
        self._x_pos = x_pos
        self._y_pos = y_pos
        self._word = word
        self._need = need

    @property
    def line_id(self):
        return self._line_id

    @property
    def x_pos(self):
        return self._x_pos

    @property
    def y_pos(self):
        return self._y_pos

    @property
    def word(self):
        return self._word

    @property
    def need(self):
        return self._need

    @line_id.setter
    def line_id(self, _line_id):
        self._line_id = _line_id

    @x_pos.setter
    def x_pos(self, _x_pos):
        self._x_pos = _x_pos

    @y_pos.setter
    def y_pos(self, _y_pos):
        self._y_pos = _y_pos

    @need.setter
    def need(self, _need):
        self._need = _need


def _extract_middle_content(_line_id, _line_list, _list_result):
    """
    夹在两个抽取字符之间的作为抽取的内容
    """
    for i in range(1, len(_line_list) - 1):
        prev_word = _line_list[i - 1]
        curr_word = _line_list[i]
        next_word = _line_list[i + 1]
        if not curr_word.need and prev_word.need and next_word.need:
            _list_result.append(Word(_line_id, curr_word.x_pos, curr_word.y_pos, curr_word.word))


def get_ocr_by_pos(ocr_data, pos_label, im_shape):
    if 'pages' not in ocr_data or len(ocr_data['pages']) != 1:
        return ""
    list_result = []
    line_id = 0

    for table in ocr_data['pages'][0]['table']:
        if not table['type']:
            for line in table['lines']:
                line_id = _parse_ocr_file(im_shape, line, line_id, list_result, pos_label)
        else:
            for cell in table['form_blocks']:
                for line in cell['lines']:
                    line_id = _parse_ocr_file(im_shape, line, line_id, list_result, pos_label)

    return list_result


def _parse_ocr_file(im_shape, line, line_id, list_result, pos_label):
    line_list = []
    for ch, ch_pos in zip(line['text'], line['char_polygons']):
        pos_list = []
        for i in range(0, len(ch_pos), 2):
            x = int(ch_pos[i])
            y = int(ch_pos[i + 1])
            pos_list.append([x, y])
        x, y, w, h = cv2.boundingRect(np.array(pos_list))
        x_label, y_label, w_label, h_label = cv2.boundingRect(np.array(pos_label))
        x_left = max(x, x_label)
        x_right = min(x + w, x_label + w_label)
        y_up = max(y, y_label)
        y_down = min(y + h, y_label + h_label)
        w_ins = x_right - x_left
        h_ins = y_down - y_up
        if w_ins > 0.6 * min(w, w_label) + 1 and h_ins > 0.6 * min(h, h_label + 1):
            (w_ins, h_ins) = get2PolyInterSectAreaSize(im_shape, pos_list, pos_label)
            if w_ins > 0.6 * min(w, w_label) + 1 and h_ins > 0.6 * min(h, h_label + 1):
                list_result.append(Word(line_id, x, y, ch))
                line_list.append(Word(x_pos=x, y_pos=y, word=ch, need=True))
            else:
                line_list.append(Word(x_pos=x, y_pos=y, word=ch, need=False))
        else:
            line_list.append(Word(x_pos=x, y_pos=y, word=ch, need=False))
    _extract_middle_content(line_id, line_list, list_result)
    return line_id + 1


def read_ner_label(ocr_data, label_data):
    list_label_result = []
    im_shape = (label_data['imageHeight'], label_data['imageWidth'])
    for label in label_data['shapes']:
        label_name = label['label']
        list_pt = []
        for pt in label['points']:
            list_pt.append([int(pt[0]), int(pt[1])])
            # print(list_pt)
        if len(list_pt) == 2:
            pt_fix = [[min(list_pt[0][0], list_pt[1][0]), min(list_pt[0][1], list_pt[1][1])],
                      [max(list_pt[0][0], list_pt[1][0]), min(list_pt[0][1], list_pt[1][1])],
                      [max(list_pt[0][0], list_pt[1][0]), max(list_pt[0][1], list_pt[1][1])],
                      [min(list_pt[0][0], list_pt[1][0]), max(list_pt[0][1], list_pt[1][1])]]
            list_pt = pt_fix
        x, y, w, h = cv2.boundingRect(np.array(list_pt))
        if w <= 5 or h <= 5:
            continue
        word_list = get_ocr_by_pos(ocr_data, list_pt, im_shape)
        list_label_result.append((label_name, word_list))
    return list_label_result
