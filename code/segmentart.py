#!/usr/bin/env python3
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt

import tools
import splitters
import representations
import re
import sys

import os
from pathos.multiprocessing import ProcessingPool as Pool

src_ext = ".aligned.pckl.txt"
ref_ext = src_ext + ".seg.ref"
alemi_ext = src_ext + ".alemiseg"
result_ext = alemi_ext + ".results"

VECTORS_PATH = "/home/pltrdy/autoalign/autoalign/legacy/word2vec_fr/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.txt"
SEPARATOR_STR = "=========="


def is_separator(line):
    return line.startswith(SEPARATOR_STR)


def seg_count(path):
    # starts at -1 since there's both begin/end separator
    count = -1
    with open(path) as f:
        for line in f:
            if is_separator(line):
                count += 1
    return count


def load_vectors(vecs_path=VECTORS_PATH):
    words = {}
    vecs = []
    with open(vecs_path) as f:
        first = True
        count = 0
        for line in f:
            if first:
                first = False
                continue
            line = line.strip()
            elements = line.split()
            word = elements[0]
            try:
                v = np.array([float(_) for _ in elements[1:]])
            except Exception as e:
                print("Excluded #%d, %s" % (count, str(e)))
                continue

            words[word] = count
            vecs.append(v)
            count += 1
    # vecs = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vecs.npy")
    # words = np.load("/home/aaa244/storage/arxiv_glove/bigrun/data/mats/vocab.npy")

    word_lookup = {w: c for c, w in enumerate(words)}
    return vecs, word_lookup


def run_folder(root_path, n_thread):
    sources = [
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.endswith(src_ext)
    ]

    vecs, word_lookup = load_vectors()
    args_list = [(src_path, vecs, word_lookup,)
                 for src_path in sources]
    with Pool(processes=n_thread) as pool:
        pool.map(run_path_args, args_list)


def run_path_args(args):
    return run_path(*args)


def run_path(src_path, vecs, word_lookup):
    ref_path = src_path.replace(src_ext, ref_ext)
    out_path = src_path.replace(src_ext, alemi_ext)

    if os.path.exists(out_path):
        print("Aborting, file exists: '%s'" % out_path)
        return
    seg_n = seg_count(ref_path)

    k = seg_n - 1
    do_run(k, src_path, out_path, vecs, word_lookup)


def do_run(K, infile, out_path, vecs, word_lookup):

    with open(infile, "r") as f:
        txt = f.read()

    punctuation_pat = re.compile(
        r"""([!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~])""")
    hyphenline_pat = re.compile(r"-\s*\n\s*")
    multiwhite_pat = re.compile(r"\s+")
    cid_pat = re.compile(r"\(cid:\d+\)")
    nonlet = re.compile(r"([^A-Za-z0-9 ])")

    def clean_text(txt):
        #txt = txt.decode("utf-8")
        txt = txt

        txt = txt.lower()
        txt = cid_pat.sub(" UNK ", txt)
        txt = hyphenline_pat.sub("", txt)
        # print punctuation_pat.findall(txt)
        txt = punctuation_pat.sub(r" \1 ", txt)
        txt = re.sub("\n", " NL ", txt)
        txt = nonlet.sub(r" \1 ", txt)

        # txt = punctuation_pat.sub(r"", txt)
        # txt = nonlet.sub(r"", txt)

        txt = multiwhite_pat.sub(" ", txt)
        txt = txt  # .encode('utf-8')
        return "".join(["START ", txt.strip(), " END"])

    txt = clean_text(txt).split()

    print(("article length:", len(txt)))

    X = []

    mapper = {}
    count = 0
    for i, word in enumerate(txt):
        if word in word_lookup:
            mapper[i] = count
            count += 1
            X.append(vecs[word_lookup[word]])

    mapperr = {v: k for k, v in list(mapper.items())}

    X = np.array(X)
    print(("X length:", X.shape[0]))

    sig = splitters.gensig_model(X)
    print("Splitting...")
    splits, e = splitters.greedysplit(X.shape[0], K, sig)
    print(splits)
    print("Refining...")
    splitsr = splitters.refine(splits, sig, 20)
    print(splitsr)

    print("Printing refined splits... ")

    for i, s in enumerate(splitsr[:-1]):
        k = mapperr[s]
        print()
        print((i, s))
        print((" ".join(txt[k-100:k]), "\n\n", " ".join(txt[k:k+100])))

    with open(out_path, "w") as f:
        prev = 0
        for s in splitsr:
            k = mapperr.get(s, len(txt))
            f.write(" ".join(txt[prev:k]).replace("NL", "\n"))
            f.write("\n%s\n" % SEPARATOR_STR)
            prev = k

    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("K", type=int)
    # parser.add_argument("infile")
    parser.add_argument("root")
    parser.add_argument("-n_thread", type=int, default=4)
    #
    args = parser.parse_args()

    run_folder(args.root, n_thread=args.n_thread)
