# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from critic.critic import gpt2_critic


def escape_for_latex(text: str):
    return text.replace("%", "\%")\
        .replace("$", "\$")\
        .replace("#", "\#").replace("&", "\&").replace("'", "\'")\
        .replace("`","\`").replace("|","\|")


if __name__ == '__main__':
    ocr_dir = Path('/home/std2020/xuqinkun/data/sroie/sroie_fixed/splits/dataset')
    ocr_file = ocr_dir / 'bad_good_pairs.txt'
    text = ocr_file.read_text()
    lines = text.split("\n")
    good_metrics = []
    bad_metrics = []
    good_better_than_bad_samples = []
    bad_better_than_good_samples = []

    ids = [i for i in range(len(lines))]
    shuffle(ids)
    max_sample_size = 600
    selected_lines = [lines[i] for i in ids[:max_sample_size]]
    for line in tqdm(selected_lines):
        good_sent, bad_sent = line.split("\t")
        good = gpt2_critic(good_sent, verbose=0)
        bad = gpt2_critic(bad_sent, verbose=0)

        if good is None or bad is None:
            continue

        prob_good, prob_bad = good[1], bad[1]
        good_sent = escape_for_latex(good_sent)
        bad_sent = escape_for_latex(bad_sent)
        pair = f"{good_sent} &{bad_sent} \\\\"
        if prob_good > prob_bad:
            good_better_than_bad_samples.append(pair)
        else:
            bad_better_than_good_samples.append(pair)
        good_metrics.append(prob_good)
        bad_metrics.append(prob_bad)

    print(f"p(bad)<p(good): {100*len(good_better_than_bad_samples)/max_sample_size:.2f}%")
    print("good\tbad\n****p(bad)<p(good)****\n")
    print("\n".join(good_better_than_bad_samples[:20]))
    print("\n****p(bad) >= p(good)****\ngood\tbad\n")
    print("\n".join(bad_better_than_good_samples[:20]))
    bins = np.arange(-200, 1, 10)
    plt.hist(bad_metrics, bins, alpha=0.5, label='Good Sample')
    plt.hist(good_metrics, bins, alpha=0.5, label='Counterpart Bad Sample')

    # plt.title('Probability')
    plt.ylabel('Density')
    plt.xlabel('log p(x)')
    plt.legend(loc='upper left')
    plt.savefig("/home/std2020/xuqinkun/matplot/pictures/good_bad_pair.svg", format="svg", )
    plt.show()