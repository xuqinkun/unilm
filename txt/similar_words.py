# -*- coding: utf-8 -*-
from pathlib import Path

w = Path("./ChineseSimilarWords.txt")
lines = w.read_text().split("\n")
group = {}
for line in lines:
    if line.strip() != '':
        words = set(line.split(","))
        if '' in words:
            words.remove('')
        for word in words:
            pool = words.copy()
            pool.remove(word)
            if word not in group:
                group[word] = pool
            else:
                group[word].update(pool)