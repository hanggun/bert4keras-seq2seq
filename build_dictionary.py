#!/usr/bin/env python3

from collections import OrderedDict
import fileinput
import sys

import numpy
import json


def main():
    word_freqs = OrderedDict()
    for filename in sys.argv[1:]:
        print('Processing', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['[PAD]'] = 0
    worddict['[UNK]'] = 1
    worddict['[CLS]'] = 2
    worddict['[SEP]'] = 3
    worddict['[MASK]'] = 4
    # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii+3

    # The JSON RFC requires that JSON text be represented using either
    # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
    # We use UTF-8 regardless of the user's locale settings.
    with open('%s.txt'%sys.argv[1], 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(worddict.keys())))

    print('Done')

if __name__ == '__main__':
    main()