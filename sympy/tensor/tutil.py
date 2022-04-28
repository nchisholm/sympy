#!/usr/bin/env python3

def partition(p, iterable):
    l_true = []
    l_false = []
    for elem in iterable:
        if p(elem):
            l_true.append(elem)
        else:
            l_false.append(elem)
    return (l_true, l_false)
