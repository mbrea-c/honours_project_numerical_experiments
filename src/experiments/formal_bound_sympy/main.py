import logging
import sympy
from sympy.printing.latex import LatexPrinter

indices_gen = sympy.numbered_symbols("i")


def layer(
    n,
    m,
    input_symbols,
    activation,
    weight_prefix="w",
    bias_prefix=None,
    indices=None,
):
    weights = sympy.MatrixSymbol(weight_prefix, m, n)
    if bias_prefix:
        bias = sympy.MatrixSymbol(bias_prefix, m, 1)
    else:
        bias = None
    if indices:
        j = indices
    else:
        j = next(indices_gen)
    i = next(indices_gen)
    z = next(indices_gen)
    output_symbols = sympy.FunctionMatrix(
        m,
        1,
        sympy.Lambda(
            (i, z),
            activation(
                sympy.summation(weights[i, j] * input_symbols[j, 0], (j, 0, n))
                + (bias[i, 0] if bias_prefix else 0)
            ),
        ),
    )
    return output_symbols


def grad(e, wrt, indices=None):
    if indices:
        i, j = indices
    else:
        i = next(indices_gen)
        j = next(indices_gen)
    dim, _ = wrt.shape
    return sympy.FunctionMatrix(dim, 1, sympy.Lambda((i, j), sympy.diff(e, wrt[i, 0])))


def norm2(vec, indices=None):
    if indices:
        i = indices
    else:
        i = next(indices_gen)
    m, _ = vec.shape
    return replace_square(sympy.summation(sympy.simplify(vec[i, 0]) ** 2, (i, 0, m)))


def replace_square(e):
    matcher = (
        lambda e: isinstance(e, sympy.Pow)
        and isinstance(e.base, sympy.Sum)
        and e.exp == 2
    )

    replacer = lambda e: _make_sum_double(e.base)

    return e.replace(matcher, replacer)


def _make_sum_double(sum):
    limit = sum.limits[0]
    i, a, b = limit
    j = next(indices_gen)
    func = sum.function
    func_2 = func.xreplace({i: j})
    return sympy.summation(sympy.summation(func * func_2, (i, a, b)), (j, a, b))


class MyLatexPrinter(LatexPrinter):
    def _print_Subs(self, expr):
        if isinstance(expr.expr, sympy.Derivative):
            deriv = expr.expr
            differand, *(wrt_counts) = deriv.args
            if len(wrt_counts) > 1 or wrt_counts[0][1] != 1:
                raise NotImplementedError("More code needed...")
            ((wrt, count),) = wrt_counts
            return f"{type(differand)}'{self._print(expr.point)}"
        else:
            return super()._print(expr)


def single_layer_no_bias():
    n = sympy.Dummy("n")
    input = sympy.MatrixSymbol("x", n, 1)
    n1 = sympy.Dummy("n1")
    m = sympy.Dummy("m")
    g = sympy.Function("g")
    t = sympy.Dummy("t")
    id = sympy.Lambda(t, t)

    out = layer(n, n1, input, g, weight_prefix="theta^1", bias_prefix=None)
    out = layer(n1, m, out, id, weight_prefix="theta^2", bias_prefix=None)

    return out, input


def two_layer_no_bias():
    n = sympy.Dummy("n")
    input = sympy.MatrixSymbol("x", n, 1)
    n1 = sympy.Dummy("n1")
    n2 = sympy.Dummy("n2")
    m = sympy.Dummy("m")
    g = sympy.Function("g")
    t = sympy.Dummy("t")
    id = sympy.Lambda(t, t)

    out = layer(n, n1, input, g, weight_prefix="theta^1", bias_prefix=None)
    out = layer(n1, n2, out, g, weight_prefix="theta^2", bias_prefix=None)
    out = layer(n2, m, out, id, weight_prefix="theta^3", bias_prefix=None)

    sympy.pprint(out[0].diff(input[0]))
    print(out[0].diff(input[0]))

    return out, input


def diff_grad_norm(out, input):
    j = sympy.Dummy("j")
    k = sympy.Dummy("k")
    diff_grad = grad(out[j, 0], wrt=input) - grad(out[k, 0], wrt=input)
    norm = norm2(diff_grad, indices=sympy.Dummy("l"))
    return norm


def run():
    sympy.init_printing(use_unicode=True)
    # out = layer(n2, m, out, id, weight_prefix="theta^3")

    logging.info(f"Computing single layer no bias")
    sl_nb = diff_grad_norm(*single_layer_no_bias())
    logging.info(f"Computing two layer no bias")
    tl_nb = diff_grad_norm(*two_layer_no_bias())

    printer = MyLatexPrinter()
    logging.info(f"Printing single layer no bias")
    with open("slnb.tex", "w") as file:
        file.write(printer.doprint(sl_nb))
    logging.info(f"Printing two layer no bias")
    with open("tlnb.tex", "w") as file:
        file.write(printer.doprint(tl_nb))
    logging.info(f"All done")
