from functools import reduce
from itertools import starmap
from sympy import permutedims
from sympy.core import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul, TensorIndex
from .tutil import replace_topdown


class PartialDerivative(TensExpr):
    """
    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    """

    def __new__(cls, expr, *variables, evaluate=True):

        expr = _sympify(expr)
        variables = tuple(_sympify(v) for v in variables)

        if not isinstance(expr, (TensExpr, Expr)):
            ValueError("Cannot differentiate %s" % repr(expr))

        # Make sure we can take a partial derivative WRT each of the variables
        for v in variables:
            if isinstance(v, Tensor) or v._diff_wrt:
                continue
            msg = ("Cannot take partial derivative with respect to %s" % v)
            raise ValueError(msg)


        # Flatten:
        if isinstance(expr, PartialDerivative):
            variables = expr.variables + variables
            expr = expr.expr

        # No-op
        if len(variables) == 0:
            return expr

        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables, replace_indices=evaluate)

        obj = TensExpr.__new__(cls, *args)

        obj._indices = indices
        obj._free = free
        obj._dum = dum

        return obj

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    @property
    def free(self):
        return self._free

    @property
    def dum(self):
        return self._dum

    @property
    def indices(self):
        return self._indices

    @property
    def expr(self):
        return self.args[0]

    # NOTE: first element of self.variables is the *innermost* partial
    # derivative (taken first).
    @property
    def variables(self):
        return self.args[1:]

    @property
    def rank(self):
        return len(self._free)

    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables,
                                         replace_indices=True):
        variables_opposite_valence = []

        for i in variables:
            if isinstance(i, Tensor):
                i_free_indices = i.get_free_indices()
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            elif isinstance(i, Symbol):
                variables_opposite_valence.append(i)

        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence,
            replace_indices=replace_indices)

        for i in range(1, len(args)):
            args_i = args[i]
            if isinstance(args_i, Tensor):
                i_indices = args[i].get_free_indices()
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        return args, indices, free, dum

    def _expand_partial_derivative(self):
        args, indices, free, dum = self._contract_indices_for_derivative(
            self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        result = obj

        if not args[0].free_symbols:
            return S.Zero
        elif isinstance(obj.expr, TensAdd):
            # take care of sums of multi PDs
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        elif isinstance(obj.expr, TensMul):
            # take care of products of multi PDs
            if len(obj.variables) == 1:
                # derivative with respect to single variable
                terms = []
                mulargs = list(obj.expr.args)
                for ind in range(len(mulargs)):
                    if not isinstance(sympify(mulargs[ind]), Number):
                        # a number coefficient is not considered for
                        # expansion of PartialDerivative
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                result = TensAdd.fromiter(terms)
            else:
                # derivative with respect to multiple variables
                # decompose:
                # partial(expr, (u, v))
                # = partial(partial(expr, u).doit(), v).doit()
                result = obj.expr  # init with expr
                for v in obj.variables:
                    result = self.func(result, v)._expand_partial_derivative()
                    # then throw PD on it

        return result

    def doit(self, deep=False):
        # TODO: deep doit=true by default
        # if deep:
        #     obj = PartialDerivative(*(arg.doit(deep=True) for arg in self.args))
        #     if not isinstance(obj, PartialDerivative):
        #         return obj
        return self._perform_derivative() #.doit()

    def _perform_derivative(self):
        # Perform iterated differentiation WRT each of the variables
        #
        # result = self.expr
        # for v in self.variables:
        #     result = _eval_partial_derivative(result, v)
        # return result

        return reduce(_eval_partial_derivative, reversed(self.variables), self.expr)

    def _eval_partial_derivative(self, v):
        v0, *vs = self.variables
        if v0 is v:
            dexpr_dv = self._eval_partial_derivative(v)
            return self.func(dexpr_dv, *vs)
        raise NotImplementedError("No assumption of equality of mixed partial derivatives")

    def get_indices(self):
        return self._indices

    def get_free_indices(self):
        free = sorted(self._free, key=lambda x: x[1])
        return [i[0] for i in free]

    def _replace_indices(self, repl):
        expr = self.expr.xreplace(repl)
        mirrored = {-k: -v for k, v in repl.items()}
        variables = [i.xreplace(mirrored) for i in self.variables]
        return self.func(expr, *variables)

    def _extract_data(self, replacement_dict):
        from .array import derive_by_array, tensorcontraction
        indices, array = self.expr._extract_data(replacement_dict)
        for variable in self.variables:
            var_indices, var_array = variable._extract_data(replacement_dict)
            var_indices = [-i for i in var_indices]
            coeff_array, var_array = zip(*[i.as_coeff_Mul() for i in var_array])
            dim_before = len(array.shape)
            array = derive_by_array(array, var_array)
            dim_after = len(array.shape)
            dim_increase = dim_after - dim_before
            array = permutedims(array, [i + dim_increase for i in range(dim_before)] + list(range(dim_increase)))
            array = array.as_mutable()
            varindex = var_indices[0]
            # Remove coefficients of base vector:
            coeff_index = [0] + [slice(None) for i in range(len(indices))]
            for i, coeff in enumerate(coeff_array):
                coeff_index[0] = i
                array[tuple(coeff_index)] /= coeff
            if -varindex in indices:
                pos = indices.index(-varindex)
                array = tensorcontraction(array, (0, pos+1))
                indices.pop(pos)
            else:
                indices.append(varindex)
        return indices, array


# TODO: remove, should be handled by using PartialDerivative.doit() now.
def perform_derivatives(expr):
    from sympy.core import Derivative

    "Evaluate (partial) derivatives in an expression (shallow)."
    if expr.args == ():
        return expr
    if isinstance(expr, PartialDerivative):
        return expr._perform_derivative()
    if isinstance(expr, Derivative):
        return expr.doit()
    return expr.func(*map(perform_derivatives, expr.args))


def _eval_partial_derivative(expr, x):
    # "Dispatch" on whether some tensor instance is being differentiated or some
    # other (scalar) expression.  We assume it makes sense to differentate with
    # respect to x and do not check otherwise.
    if isinstance(expr, TensExpr):
        return expr._eval_partial_derivative(x)
    if isinstance(x, TensExpr):
        # expr is a non-`TensExpr` type
        if isinstance(x, Tensor):
            if expr.args == ():
                # expr and x cannot be the same b/c they have different types
                return S.Zero
            return _chaindiff(expr, x)
        raise ValueError("Cannot differentiate with respect to expression %s"
                         % repr(x))
    return expr._eval_derivative(x)  # left with ordinary, non-tensor symbols


def _chaindiff(expr, x):
    # Multivariate chain rule on `Expr`s that contain (zero-order) `TensExpr`s

    # TODO: need to ensure dummy indices in `expr` are different than dummy
    # indices in `x`.
    #   + Know that `expr` has all self-contained dummy indices
    #   + Might get a dummy collision from xs

    def dummify_scalar_tensexpr(tex: TensExpr):
        if isinstance(tex, TensExpr):
            if tex.rank == 0:
                return Dummy()
            else:
                raise ValueError("Non-scalar TensExpr nested in a plain Expr")
        return None

    expr_dumb, dumb_exprs = replace_topdown(dummify_scalar_tensexpr, expr)
    # expr_dumb: `expr` with nested `TensExpr`s replaced with dummy symbols
    # dumb_exprs: `dict` mapping each dummy symbol to the `TensExpr` it replaces

    def expand_term(dummy_var, scalar_tensexpr):

        # FIXME: handle index naming collisions
        # cdt = _count_dum_pair_types(scalar_tensexpr)
        # new_inds = list(x.indices)
        # dumset = scalar_tensexpr._get_dummy_indices_set()
        # for ind in x.indices:
        #     if ind in dumset:
        #         ind_type = ind.tensor_index_type
        #         new_ind_name = ind_type.dummy_name + "_" + str(cdt[ind_type])
        #         new_ind = TensorIndex(new_ind_name, ind_type, ind.is_up)
        #         new_inds.append(new_ind)
        #         cdt[ind_type] += 1
        #     else:
        #         new_inds.append(ind)

        # x1 = x.head(*new_inds)

        # [@F(..., uᵢ, ...)/@uᵢ]_{uᵢ = ϕᵢ(𝐱)} * @ϕᵢ(𝐱)/@xₖ
        deriv_dummy = _eval_partial_derivative(expr_dumb, dummy_var)
        partial_deriv = scalar_tensexpr._eval_partial_derivative(x)
        return TensMul(deriv_dummy, partial_deriv).doit(deep=False)

    terms = tuple(starmap(expand_term, dumb_exprs.items()))

    return TensAdd.fromiter(terms).doit(deep=False).xreplace(dumb_exprs)



from collections import Counter, defaultdict
from sympy.core.traversal import preorder_traversal

# def _count_dum_pair_types(expr):
#     """Count how many dummy pairs of each type there are in `expr`.
#
#     Returns a `defaultdict(int)` mapping each index type to the maximum number
#     of occurances.
#     """
#
#     counter = defaultdict(int)
#
#     for _expr in preorder_traversal(expr):
#
#         if not isinstance(_expr, TensExpr) or isinstance(_expr, TensAdd):
#             continue
#
#         inds = _expr.get_indices()
#
#         c = Counter(inds[copos].tensor_index_type for (copos, _) in _expr.dum)
#
#         for (index_type, count) in c.items():
#             count0 = counter[index_type]
#             counter[index_type] = max(count, count0)
#
#     return counter


def _count_dum_pair_types(expr, _counter=None):
    """Count how many dummy pairs of each type there are in `expr`.

    Returns a `defaultdict(int)` mapping each index type to the maximum number
    of occurances.
    """

    _counter = defaultdict(int) if _counter == None else _counter

    if isinstance(expr, TensExpr) and not isinstance(expr, TensAdd):
        inds = expr.get_indices()
        return Counter(inds[copos].tensor_index_type for (copos, _) in expr.dum)


    for arg in expr.args:
        c = _count_dum_pair_types(arg, _counter)
        _counter.update({index_type: max(count, _counter[index_type])
                         for (index_type, count) in c.items()})

    return _counter
