"""Exact arithmetic over an expression the speaker wrote - §11.76.

The system could combine two STORED numbers ("the sum of the tower
height and the bridge height") since §11.42, but it could not add two.
"what is 2 + 2?" fell all the way through the question ladder to "I
don't hold anything on that yet" - honest, and useless. This module
reads an arithmetic expression and evaluates it.

**It evaluates over exact rationals, not floats.** That is the whole
design choice, and it is not decoration: the obvious implementation
answers "what is 0.1 + 0.2?" with 0.30000000000000004, because binary
floating point cannot hold a tenth. A system whose entire discipline
is "say exactly what is true" cannot answer a question about tenths
with a number that is not the answer. Every value here is a
``Fraction``; decimals are parsed exactly (``Fraction("0.1")`` is one
tenth, not the nearest double); and the renderer prints an exact
decimal when the denominator allows one and an exact fraction when it
does not - "1/3", never "0.3333333333333333", which is a different
number.

The grammar is ordinary and complete for its class::

    expr  := term (('+' | '-') term)*
    term  := unary (('*' | '/') unary)*
    unary := ('-' | '+')* atom
    atom  := NUMBER | '(' expr ')'

so precedence and parentheses are structural, not special-cased, and
"3 + 4 * 5" is 23 while "(3 + 4) * 5" is 35. Word operators are the
same operators: plus, minus, times, multiplied by, divided by, over.

Two lines hold, in the house's tradition. Division by zero REFUSES
aloud - it is undefined, and a calculator that returns something for
it is lying about arithmetic itself. And the reader is strict: an
expression with any token that is not a number, an operator, or a
parenthesis is not an arithmetic question at all, and this module
returns None so the rest of the ladder can answer it. A math branch
that swallowed "what is the tower height?" would be a worse bug than
having no math.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

__all__ = ["MathResult", "evaluate", "render_exact", "strip_question"]

#: Word operators, mapped to their symbols. "divided"/"multiplied"
#: swallow a following "by"; "over" is division as spoken.
_WORD_OPS = {
    "plus": "+", "minus": "-", "times": "*", "over": "/",
    "divided": "/", "multiplied": "*",
}

#: Openers stripped from a question before reading the expression.
_PREFIXES = (
    "what is", "what's", "whats", "what does", "how much is",
    "calculate", "compute", "evaluate", "work out", "tell me",
    "equals", "equal",
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[()+\-*/]|[a-z']+")

#: When False the module evaluates in binary floating point - the
#: standard option, implemented honestly, and the gate's baseline arm.
_EXACT_RATIONALS = True


@dataclass
class MathResult:
    """One evaluated expression, or one refusal.

    Attributes:
        shown: The rendered value, exact.
        refusal: Why nothing was computed, if nothing was.
        expression: The expression as it was read back, normalised.
        fractional: The value has no exact decimal form, so the
            answer is a fraction - worth saying, because a reader
            who expected a decimal should know none exists.
    """

    shown: str = ""
    refusal: str = ""
    expression: str = ""
    fractional: bool = False

    @property
    def answered(self) -> bool:
        return not self.refusal and self.shown != ""


def strip_question(text: str) -> str:
    """Remove a question's opener and its punctuation."""
    low = text.lower().strip().strip("?!. ")
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if low.startswith(prefix + " "):
                low = low[len(prefix) + 1:].strip()
                changed = True
            elif low == prefix:
                return ""
    return low


def _tokenize(text: str) -> list[str] | None:
    """Symbols and numbers, or None if anything else is in the way.

    Strictness is the point: this decides whether the question is an
    arithmetic question at all, and a false yes eats a question some
    other branch could have answered.
    """
    raw = _TOKEN_RE.findall(text)
    if not raw:
        return None
    if "".join(raw) != re.sub(r"\s+", "", text):
        # Something in the text was not tokenizable (a stray symbol,
        # a unit, a word with digits in it) - not our question.
        return None
    tokens: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if _NUMBER_RE.fullmatch(token) or token in "()+-*/":
            tokens.append(token)
        elif token in _WORD_OPS:
            tokens.append(_WORD_OPS[token])
            if (token in ("divided", "multiplied")
                    and index + 1 < len(raw) and raw[index + 1] == "by"):
                index += 1
        else:
            return None
        index += 1
    if not any(_NUMBER_RE.fullmatch(tok) for tok in tokens):
        return None
    return tokens


class _Reader:
    """Recursive descent over the token list."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.at = 0

    def peek(self) -> str | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expr(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            right = self.term()
            value = value + right if op == "+" else value - right
        return value

    def term(self):
        value = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()
            right = self.unary()
            if op == "*":
                value = value * right
            else:
                if right == 0:
                    raise ZeroDivisionError
                value = value / right
        return value

    def unary(self):
        if self.peek() in ("+", "-"):
            op = self.take()
            value = self.unary()
            return -value if op == "-" else value
        return self.atom()

    def atom(self):
        token = self.peek()
        if token is None:
            raise SyntaxError("expression ends early")
        if token == "(":
            self.take()
            value = self.expr()
            if self.peek() != ")":
                raise SyntaxError("unclosed parenthesis")
            self.take()
            return value
        if _NUMBER_RE.fullmatch(token):
            self.take()
            return Fraction(token) if _EXACT_RATIONALS else float(token)
        raise SyntaxError(f"unexpected {token!r}")


def render_exact(value) -> str:
    """Print a rational exactly: integer, decimal, or fraction.

    A denominator whose only prime factors are 2 and 5 has an exact
    decimal form, and it is printed in full. Any other denominator has
    none - one third is not 0.333333, and printing that instead would
    answer a question about thirds with a number that is not the
    answer - so the fraction itself is the answer.
    """
    if not isinstance(value, Fraction):
        # The float arm: render as the standard option would.
        if float(value).is_integer():
            return str(int(value))
        return repr(float(value))
    if value.denominator == 1:
        return str(value.numerator)
    rest = value.denominator
    twos = fives = 0
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(twos, fives)
    scaled = value.numerator * (10 ** places) // value.denominator
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _echo(tokens: list[str]) -> str:
    """The expression as it was READ, spaced the way people write.

    Echoing the reading is what makes precedence checkable at a
    glance: "3 + 4 * 5 = 23" shows which multiplication went first.
    A parenthesis hugs its contents and a sign hugs its number, so
    the echo looks like the arithmetic it is.
    """
    out = ""
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else None
        hugs = (not out or previous == "(" or token == ")"
                or _is_unary_at(tokens, index - 1))
        out += token if hugs else " " + token
    return out


def _is_unary_at(tokens: list[str], index: int) -> bool:
    """Whether the operator at ``index`` is a sign, not a joiner."""
    if index < 0 or tokens[index] not in ("+", "-"):
        return False
    return index == 0 or tokens[index - 1] in "(+-*/"


def evaluate(text: str) -> MathResult | None:
    """Evaluate an arithmetic question, or None if it is not one.

    None means "not my question" and is the common case; a MathResult
    with a refusal means "an arithmetic question with no answer",
    which is a different thing and is said aloud.
    """
    stripped = strip_question(text)
    if not stripped:
        return None
    tokens = _tokenize(stripped)
    if tokens is None:
        return None
    if not any(tok in "+-*/" for tok in tokens):
        # A bare number is not a calculation, and answering "5" to
        # "what is 5?" would be a parlour trick, not arithmetic.
        return None
    reader = _Reader(tokens)
    try:
        value = reader.expr()
    except ZeroDivisionError:
        return MathResult(
            refusal="division by zero is undefined - there is no "
                    "number to give you",
            expression=_echo(tokens))
    except (SyntaxError, IndexError):
        return None
    if reader.at != len(reader.tokens):
        return None
    shown = render_exact(value)
    return MathResult(shown=shown, expression=_echo(tokens),
                      fractional="/" in shown)
