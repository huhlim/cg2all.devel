#!/usr/bin/env python

import random

n_test = 720
n_valid = 720


def read_list(fn):
    targets = []
    with open(fn) as fp:
        for line in fp:
            if not line.startswith("#"):
                targets.append(line.strip())
    return set(targets)


def write_list(fn, set):
    with open(fn, "wt") as fout:
        for tg in set:
            fout.write(f"{tg}\n")


def main():
    pdb_s = {}
    for size in ["6k", "29k"]:
        pdb_s[size] = read_list(f"pdb.{size}/targets")
    #
    pdb_common = list(pdb_s["6k"].intersection(pdb_s["29k"]))
    #
    random.shuffle(pdb_common)
    #
    test_set = set(pdb_common[:n_test])
    valid_set = set(pdb_common[-n_valid:])
    #
    for size in ["6k", "29k"]:
        pdb_s[size] -= test_set
        pdb_s[size] -= valid_set
        #
        write_list(f"pdb.{size}/targets.train", pdb_s[size])
        write_list(f"pdb.{size}/targets.test", test_set)
        write_list(f"pdb.{size}/targets.valid", valid_set)


if __name__ == "__main__":
    main()
