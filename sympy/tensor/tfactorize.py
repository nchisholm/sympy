#!/usr/bin/env python3

from functools import reduce
from operator import or_

from sympy.tensor.tensor import TensMul


def merge_intersecting0(set0, others):
    """
    Merge `set0` with `others` that intersect it and return the result.
    Disjoint sets are left alone and returned.
    """
    i = 0
    while i < len(others):
        s = others[i]
        if set0 & s != 0:
            set0 |= s
            others = others[:i] + others[i+1:]
            i = 0
        else:
            i += 1
    return (set0, others)  # no merge possible


def merge_intersecting(sets):

    sets_out = []

    while len(sets) > 0:
        s = sets[0]
        s1, sets = merge_intersecting0(s, sets[1:])
        sets_out.append(s1)

    return sets_out


def bits2int(bit_positions):
    """Return the integer with 1's at positions `bitseq`.

    Position indices start at the least significant bit.

    See also int2bits.
    """
    val = 0
    for i in bit_positions:
        val |= 1 << i
    return val


def int2bits(val: int, length=None):
    """Give the positions of the bits, from least to most significant, of the
    integer val."""
    if length is None:
        length = int.bit_length(val)
    return tuple(filter(lambda i: val & 1 << i, range(length)))


# Create sets of the `args` of a `TensMul` instance that are "connected" by
# common dummy indices and return them.

# First, for each dummy index, we compute an integer whose `1` bits
# correspond the positions of the individual `Tensor`s in `tensmul` that
# participate in contraction with the dummy index.  Thus, each of these
# integers "contain" only one or two `1` bits.  For example, consider the
# following expression.
#
#      A(-i,j,-j) * x(-l) * v(m) * e(k) * y(l) * B(i, -k, n)
#
# Here, i, j, k, and l are dummy indices and m and n are free.
#
#               tensor (head)
#  free          /
#  /  A x v e y B    Integer
# m: [0 0 1 0 0 0]
# n: [0 0 0 0 0 1]
# --
# i: [1 0 0 0 0 1] = 17       (least -> most significan bit)
# j: [1 0 0 0 0 0] =  1
# k: [0 0 0 1 0 1] = 20
# l: [0 1 0 0 1 0] = 10
# --
# \
#  dummy
#
# We then iteratively merge these "integers" as sets (via logical or) of
# bits whenever they share a `1` bit corresponding to a common tensor
# factor.  The result is a set of factors

def bitrange(m, n):
    # Returns an integer whose binary representation contains ones from the
    # m-th LSB to the (n-1)-th LSB.
    return (1<<n)-1 ^ (1<<m)-1


def _factor_free(tensmul: TensMul):

    free_args = tuple(bits2int((icomp,))
                      for (_, _, icomp) in tensmul.free_in_args)

    assert tuple(sorted(free_args)) == free_args

    dummy_arg_pairs = tuple(bits2int((icomp1, icomp2))
                            for (_, _, icomp1, icomp2) in tensmul.dum_in_args)

    free_grps = []

    i = 0
    while i < len(free_args):
        s_free = free_args[i]
        s1_free, dummy_arg_pairs = merge_intersecting0(s_free, dummy_arg_pairs)
        n_args_handled = int.bit_length(s1_free) - i
        mask = bitrange(i, n_args_handled)
        free_grp = s1_free
        for _s in free_args[i+1:]:
            free_grp |= _s & mask
        free_grps.append(free_grp)
        i += n_args_handled

    return free_grps, dummy_arg_pairs


def factor_outer(tensmul: TensMul):

    tensmul_nocoeff = tensmul.nocoeff

    free_grps, dummy_arg_pairs = _factor_free(tensmul_nocoeff)
    dummy_grps = merge_intersecting(dummy_arg_pairs)

    def int2tensmul(val: int):
        args = tensmul_nocoeff.args
        return TensMul.fromiter(args[i] for i in int2bits(val))

    return (tensmul.coeff,
            tuple(map(int2tensmul, dummy_grps)),
            tuple(map(int2tensmul, free_grps)))


def factor_scalar_tensor(tensmul: TensMul):
    coeff, scalar_factors, tensor_factors = factor_outer(tensmul)
    return (TensMul(coeff, *scalar_factors), TensMul(*tensor_factors))
