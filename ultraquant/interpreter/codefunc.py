"""The code function: a sandboxed mini-Python evaluator.

The interpreter can *compute*, not merely talk.  Source is parsed to an AST and
validated against a strict node whitelist before anything runs: no imports, no
attribute access other than a fixed set of ``math`` functions, no dunder or
underscore-prefixed identifiers (which is what closes the
``().__class__.__bases__`` style of escape), and no access to real builtins.
Execution is additionally bounded by an operation budget and a wall-clock
timeout so that a runaway loop cannot wedge the session.

This is a deliberately small language, not a general Python sandbox: everything
not explicitly permitted is refused.
"""

from __future__ import annotations

import ast
import math
import sys
import threading
from typing import Any

__all__ = ["CodeError", "SafeCodeRunner"]


class CodeError(Exception):
    """Raised when code is rejected by the sandbox or fails while running."""


#: AST node types the sandbox will execute.  Anything absent is refused.
_ALLOWED_NODES: frozenset[type] = frozenset(
    {
        ast.Module,
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.Return,
        ast.Lambda,
        ast.arguments,
        ast.arg,
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.IfExp,
        ast.Call,
        ast.keyword,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.Subscript,
        ast.Slice,
        ast.Index if hasattr(ast, "Index") else ast.Slice,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.Attribute,  # gated hard in the validator: math.<whitelisted> only
        # operators / comparators / boolean ops
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.Invert,
    }
)

#: The only attributes reachable at all, and only on the ``math`` name.
_MATH_WHITELIST: frozenset[str] = frozenset(
    {
        "sqrt", "sin", "cos", "tan", "pi", "e", "log", "log2", "log10", "exp",
        "floor", "ceil", "fabs", "tanh", "atan", "atan2", "hypot", "gcd",
        "comb", "perm",
    }
)


class _MathProxy:
    """Attribute-restricted stand-in for the ``math`` module."""

    def __getattr__(self, name: str) -> Any:
        if name not in _MATH_WHITELIST:
            raise CodeError(f"math.{name} is not available in the sandbox")
        return getattr(math, name)


def _safe_namespace(out: list[str]) -> dict[str, Any]:
    """Build the execution globals: a fixed toolkit and no real builtins."""

    def _print(*args: Any, sep: str = " ", end: str = "\n") -> None:
        out.append(sep.join(str(a) for a in args) + end)

    return {
        "__builtins__": {},
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "round": round, "pow": pow, "sorted": sorted,
        "reversed": reversed, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "print": _print,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "math": _MathProxy(),
        # The Greek letters, as bare names: from-scratch special functions and
        # entropy *measurements* (quantum/symbols.py), each gated against
        # reference values. Plain callables and floats - no attribute access,
        # so they pass the validator the same way abs and len do.
        **_symbol_namespace(),
    }


def _symbol_namespace() -> dict[str, Any]:
    """The sandbox-safe slice of :mod:`ultraquant.quantum.symbols`."""
    from ultraquant.quantum import symbols

    return {
        "gamma_fn": symbols.gamma_fn, "zeta": symbols.zeta, "xi": symbols.xi,
        "eta": symbols.eta, "phi": symbols.phi, "tau": symbols.tau,
        "euler_gamma": symbols.euler_gamma, "shannon": symbols.shannon,
        "min_entropy": symbols.min_entropy, "mu": symbols.mu,
        "sigma": symbols.sigma, "rho": symbols.rho, "chi": symbols.chi,
        "nu": symbols.nu, "omega": symbols.omega,
        "efficiency": symbols.efficiency, "theta": symbols.theta,
    }


class SafeCodeRunner:
    """Validate and execute a restricted subset of Python."""

    def __init__(self, max_ops: int = 200_000, timeout_s: float = 2.0) -> None:
        """Configure the sandbox limits.

        Args:
            max_ops: Maximum traced events before execution is aborted.
            timeout_s: Wall-clock limit for a single :meth:`run` call.
        """
        self.max_ops = int(max_ops)
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------ validation

    def _validate(self, tree: ast.AST) -> None:
        """Walk the AST and refuse anything outside the whitelist.

        Raises:
            CodeError: On the first disallowed construct found.
        """
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_NODES:
                raise CodeError(f"{type(node).__name__} is not allowed in the sandbox")

            if isinstance(node, ast.Attribute):
                if not isinstance(node.ctx, ast.Load):
                    raise CodeError("attribute assignment is not allowed")
                if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                    raise CodeError("attribute access is only allowed on 'math'")
                if node.attr not in _MATH_WHITELIST:
                    raise CodeError(f"math.{node.attr} is not available in the sandbox")

            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise CodeError(f"identifier {node.id!r} is not allowed")
            if isinstance(node, ast.arg) and node.arg.startswith("_"):
                raise CodeError(f"parameter {node.arg!r} is not allowed")
            if isinstance(node, ast.keyword) and node.arg and node.arg.startswith("_"):
                raise CodeError(f"keyword {node.arg!r} is not allowed")
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_"):
                    raise CodeError(f"function {node.name!r} is not allowed")
                if node.decorator_list:
                    raise CodeError("decorators are not allowed")
                args = node.args
                if args.vararg or args.kwarg or args.posonlyargs or args.kwonlyargs:
                    raise CodeError("only plain positional parameters are allowed")

    # ------------------------------------------------------------- execution

    def run(self, source: str) -> dict:
        """Execute ``source`` and return its result, output and definitions.

        Args:
            source: The mini-Python source to run.

        Returns:
            ``{"result": <value of a trailing expression or None>,
            "stdout": <captured print output>, "defined": [names]}``.

        Raises:
            CodeError: On a syntax error, a disallowed construct, a runtime
                error, an operation-budget breach, or a timeout.
        """
        if not source.strip():
            return {"result": None, "stdout": "", "defined": []}
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise CodeError(f"syntax error: {exc.msg}") from exc

        self._validate(tree)

        # A trailing bare expression becomes the reported result.
        tail: ast.Expr | None = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = tree.body.pop()  # type: ignore[assignment]

        captured: list[str] = []
        namespace = _safe_namespace(captured)
        body_code = compile(ast.Module(body=tree.body, type_ignores=[]), "<uq-code>", "exec")
        tail_code = (
            compile(ast.Expression(body=tail.value), "<uq-code>", "eval") if tail else None
        )

        outcome: dict[str, Any] = {}
        limit = self.max_ops

        def worker() -> None:
            counter = 0

            def tracer(frame, event, arg):  # noqa: ANN001 - CPython trace protocol
                nonlocal counter
                counter += 1
                if counter > limit:
                    raise CodeError(f"operation budget of {limit} exceeded")
                return tracer

            try:
                sys.settrace(tracer)
                exec(body_code, namespace)  # noqa: S102 - sandboxed namespace
                outcome["result"] = eval(tail_code, namespace) if tail_code else None  # noqa: S307
            except CodeError as exc:
                outcome["error"] = exc
            except Exception as exc:  # noqa: BLE001 - surfaced as CodeError
                outcome["error"] = CodeError(f"{type(exc).__name__}: {exc}")
            finally:
                sys.settrace(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_s)
        if thread.is_alive():
            # The worker is daemonic and op-budgeted; it cannot outlive the
            # process and will die at the budget even if it ignores this.
            raise CodeError(f"timed out after {self.timeout_s:g}s")
        if "error" in outcome:
            raise outcome["error"]

        defined = sorted(k for k in namespace if not k.startswith("_") and k not in _safe_namespace([]))
        return {
            "result": outcome.get("result"),
            "stdout": "".join(captured),
            "defined": defined,
        }
