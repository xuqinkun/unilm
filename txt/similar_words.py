# -*- coding: utf-8 -*-
from pathlib import Path

w = Path("txt/ChineseSimilarWords.txt")
lines = w.read_text().split("\n")
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