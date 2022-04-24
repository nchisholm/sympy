#!/usr/bin/env python3

from sympy.core import S

from sympy.tensor.tensor import TensExpr, TensAdd, TensMul


def _split_sums(expr: TensMul):

    # This internal function expects a TensMul object with the coefficent
    # stripped
    assert expr.coeff is TensMul.identity

    if isinstance(expr, TensAdd):
        # Problematic for now; we need a way of to handle indices over multiple
        # terms in a TensAdd object.
        raise NotImplementedError()

    icomps_free = tuple(set((icomp,))
                       for (_, _, icomp) in expr.free_in_args)
    icomps_dummy = tuple(set((icomp1, icomp2))
                        for (_, _, icomp1, icomp2) in expr.dum_in_args)

    # Find groups of components, summations over which can be written as
    # separate factors
    icomp_sets = merge_intersecting(icomps_free + icomps_dummy)

    def tensmul_from_iargs(icomps):
        return TensMul.fromiter(expr.args[i] for i in icomps).doit()

    return tuple(map(tensmul_from_iargs, icomp_sets))

def split_sums(expr):
    if not isinstance(expr, TensMul):
        return (expr,)
    coeff = expr.coeff
    nocoeff = expr.nocoeff
    return (*(() if coeff is S.One else (coeff,)),
            *_split_sums(nocoeff))

def scalar_tensor_sums(expr):
    if not isinstance(expr, TensExpr):
        return (expr, (), ())
    coeff = expr.coeff
    nocoeff = expr.nocoeff
    factors = _split_sums(nocoeff)
    scalars, tensors = partition(lambda ex: ex.rank == 0, factors)
    return (coeff, tuple(scalars), tuple(tensors))


def merge_intersecting(sets):

    sets_out = []

    while len(sets) > 0:
        s = sets[0]
        s1, sets = merge_intersecting1(s, sets[1:])
        sets_out.append(s1)

    return sets_out


def merge_intersecting1(set0, others):
    """
    Merge `set0` with `others` that intersect it and return the result.
    Disjoint sets are left alone and returned.
    """
    i = 0
    while i < len(others):
        s = others[i]
        if not set0.isdisjoint(s):
            set0 |= s
            others = others[:i] + others[i+1:]
            i = 0
        else:
            i += 1
    return (set0, others)



def partition(p, iterable):
    l_true = []
    l_false = []
    for elem in iterable:
        if p(elem):
            l_true.append(elem)
        else:
            l_false.append(elem)
    return (l_true, l_false)
