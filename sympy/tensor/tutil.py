#!/usr/bin/env python3

def replace_topdown(f, expr):
    return _replace_topdown(f, expr, {})

def _replace_topdown(f, expr, substitutions):
    # Top to bottom replacement of subexpressions

    replacement = f(expr)
    if replacement is not None:
        substitutions[replacement] = expr
        return (replacement, substitutions)

    args0 = expr.args

    if args0 == ():
        return (expr, substitutions)

    args1 = []
    for arg in args0:
        arg1, _ = _replace_topdown(f, arg, substitutions)
        args1.append(arg1)
    return (expr.func(*args1), substitutions)


def partition(p, iterable):
    l_true = []
    l_false = []
    for elem in iterable:
        if p(elem):
            l_true.append(elem)
        else:
            l_false.append(elem)
    return (l_true, l_false)
