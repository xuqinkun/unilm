# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(__file__))
from pathlib import Path

root_dir = Path(os.path.dirname(__file__))
word_file = root_dir/"ChineseSimilarWords.txt"
lines = word_file.read_text().split("\n")
word_dict = {}
for line in lines:
    if line.strip() != '':
        words = set(line.split(","))
        if '' in words:
            words.remove('')
        for word in words:
            pool = words.copy()
            pool.remove(word)
            if word not in word_dict:
                word_dict[word] = pool
            else:
                word_dict[word].update(pool)
words = list(word_dict.keys())