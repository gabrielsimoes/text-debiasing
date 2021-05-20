"""Simple implementation of single-reference, case-sensitive BLEU
without tokenization."""

from __future__ import division
from six.moves import range, zip
from six import itervalues
import collections
import math
import nltk
import itertools

def ngrams(seg, n):
    c = collections.Counter()
    for i in range(len(seg)-n+1):
        c[tuple(seg[i:i+n])] += 1
    return c

def card(c):
    """Cardinality of a multiset."""
    return sum(itervalues(c))

def zero():
    return collections.Counter()

def count(t, r, n=4):
    """Collect statistics for a single test and reference segment."""

    stats = collections.Counter()
    for i in range(1, n+1):
        tngrams = ngrams(t, i)
        stats['guess',i] += card(tngrams)
        stats['match',i] += card(tngrams & ngrams(r, i))
    stats['reflen'] += len(r)
    return stats

def bleu_score(stats, n=4):
    """Compute BLEU score.

    :param stats: Statistics collected using bleu.count
    :type stats: dict"""

    b = 1.
    for i in range(1, n+1):
        b *= stats['match',i]/stats['guess',i] if stats['guess',i] > 0 else 0
    b **= 0.25
    if stats['guess',1] < stats['reflen']:
        b *= math.exp(1-stats['reflen']/stats['guess',1])
    return b

def download_wordnet():
    try:
        nltk.data.find('corpora/wordnet');
    except:
        nltk.download('wordnet')

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('test', metavar='predict', help='predicted translations')
    argparser.add_argument('gold', metavar='true', help='true translations')
    argparser.add_argument('-n', help='maximum n-gram size to score', default=4, type=int)
    args = argparser.parse_args()

    test_bleu = [line.split() for line in open(args.test)]
    gold_bleu = [line.split() for line in open(args.gold)]

    hits = []
    for t, g in zip(test_bleu, gold_bleu):
        for tkt, tkg in itertools.zip_longest(t, g):
            hits.append(tkt == tkg)
    print("Per-Token Accuracy:", sum(hits)/len(hits))

    hits = []
    for t, g in zip(test_bleu, gold_bleu):
        hits.append(t == g)
    print("Sentence Accuracy:", sum(hits)/len(hits))

    c = zero()
    for t, g in zip(test_bleu, gold_bleu):
        c += count(t, g, n=args.n)
    print("BLEU:", bleu_score(c, n=args.n))

    test_meteor = [line.strip() for line in open(args.test)]
    gold_meteor = [line.strip() for line in open(args.gold)]

    download_wordnet()
    s = 0
    for t, g in zip(test_meteor, gold_meteor):
        s += nltk.translate.meteor_score.meteor_score([g], t)
    s /= len(test_meteor)
    print("METEOR:", s)



