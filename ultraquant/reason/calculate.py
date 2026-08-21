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
    atom  := OPERAND | '(' expr ')'

so precedence and parentheses are structural, not special-cased, and
"3 + 4 * 5" is 23 while "(3 + 4) * 5" is 35. Word operators are the
same operators: plus, minus, times, multiplied by, divided by, over.

**An operand is one of three things** (§11.78, when a memory is
given): a written number, a written quantity ("300 meters"), or a
held belief ("the tower height"). Units ride the operators rather
than being stripped off before the arithmetic and pasted back after
- addition needs one unit and reads in the LARGER of the two, which
is the rule §11.42's combine path already answers by; multiplication
takes at most one, because meters times meters would name a unit no
definition covers; and a quantity over a quantity in one family is a
plain ratio. Conversions apply §11.42's definitions EXACTLY, read
back through their decimal spelling: 0.3048 is what a foot is in
meters, and the double nearest 0.3048 is not.

Two lines hold, in the house's tradition. Division by zero REFUSES
aloud - it is undefined, and a calculator that returns something for
it is lying about arithmetic itself. And the reader would rather
decline than guess: without a memory, any token that is not a
number, an operator or a parenthesis means "not my question"; with
one, an expression where NOTHING resolves means the same, so "what
is the plus size?" - which reads "plus" as an operator - falls
through untouched. A math branch that swallowed "what is the tower
height?" would be a worse bug than having no math.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

__all__ = ["MathResult", "Quantity", "Undefined", "evaluate",
           "render_exact", "strip_question"]


class Undefined(Exception):
    """An operation with no answer, carrying why it has none."""


class _NotArithmetic(Exception):
    """This text is not an arithmetic question after all."""


@dataclass(frozen=True)
class Quantity:
    """A number, and the unit it is a number OF ("" for a pure one).

    Units ride the arithmetic rather than being stripped off before
    it and pasted back after: that is what makes "300 meters + 2
    kilometers" answerable and "300 meters + 2" refusable, which are
    the same rule seen from two sides.
    """

    value: object
    unit: str = ""

    def __repr__(self) -> str:            # pragma: no cover - debug
        return f"Quantity({self.value}, {self.unit!r})"


def _factors(unit: str):
    """The exact conversion table for ``unit``'s family, or None.

    §11.42's table stores its factors as floats, and a float is the
    wrong shape for a definition: 0.3048 is EXACTLY what a foot is in
    meters, but the double nearest 0.3048 is not. The definitions are
    read back through their decimal spelling, which restores them
    exactly - so a conversion here is a definition applied, never an
    approximation compounded.
    """
    from ultraquant.reason.inference import (_UNIT_FAMILIES,
                                             _UNIT_TO_FAMILY)

    family = _UNIT_TO_FAMILY.get(unit)
    if family is None:
        return None
    return {name: Fraction(str(factor))
            for name, factor in _UNIT_FAMILIES[family].items()}


def _same_family(left: str, right: str) -> bool:
    from ultraquant.reason.inference import _UNIT_TO_FAMILY

    return (_UNIT_TO_FAMILY.get(left) is not None
            and _UNIT_TO_FAMILY.get(left) == _UNIT_TO_FAMILY.get(right))


def _aligned(left: Quantity, right: Quantity):
    """Both values in one unit, plus that unit - or a refusal.

    The result reads in the LARGER of the two units, which is the
    rule §11.42's combine path already answers by; one store must
    not have two voices for one arithmetic.
    """
    if left.unit == right.unit:
        return left.value, right.value, left.unit
    if not left.unit or not right.unit:
        raise Undefined(
            f"'{_show(left)}' and '{_show(right)}' do not agree on a "
            "unit, and assuming one would be invention")
    if not _same_family(left.unit, right.unit):
        raise Undefined(
            f"no definition I hold connects {left.unit}s and "
            f"{right.unit}s")
    table = _factors(left.unit)
    target = (left.unit if table[left.unit] >= table[right.unit]
              else right.unit)
    scale_l = table[left.unit] / table[target]
    scale_r = table[right.unit] / table[target]
    return left.value * scale_l, right.value * scale_r, target


def _show(quantity: Quantity) -> str:
    text = render_exact(quantity.value)
    return f"{text} {quantity.unit}s" if quantity.unit else text


def _add(left: Quantity, right: Quantity, sign: int) -> Quantity:
    a, b, unit = _aligned(left, right)
    return Quantity(a + b if sign > 0 else a - b, unit)


def _mul(left: Quantity, right: Quantity) -> Quantity:
    if left.unit and right.unit:
        raise Undefined(
            f"multiplying {left.unit}s by {right.unit}s would name a "
            "unit no definition I hold covers")
    return Quantity(left.value * right.value, left.unit or right.unit)


def _div(left: Quantity, right: Quantity) -> Quantity:
    if right.value == 0:
        raise ZeroDivisionError
    if left.unit and right.unit:
        if not _same_family(left.unit, right.unit):
            raise Undefined(
                f"no definition I hold connects {left.unit}s and "
                f"{right.unit}s")
        a, b, _unit = _aligned(left, right)
        # A length over a length is a plain ratio, and saying so is
        # more useful than refusing: "twice as long" is the answer.
        return Quantity(a / b, "")
    if right.unit:
        raise Undefined(
            f"dividing a plain number by {right.unit}s would name a "
            "unit no definition I hold covers")
    return Quantity(left.value / right.value, left.unit)

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
_SIGNED_RE = re.compile(r"-?\d+(?:\.\d+)?")
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
        premises: The held facts an operand came from, if any. A
            calculation over beliefs is a derivation, and says so.
        confidence: The least confident premise, or None when the
            arithmetic rested on nothing but written numbers.
    """

    shown: str = ""
    refusal: str = ""
    expression: str = ""
    fractional: bool = False
    premises: list = field(default_factory=list)
    confidence: float | None = None

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


def _is_unit(word: str) -> bool:
    from ultraquant.reason.inference import _UNIT_TO_FAMILY
    from ultraquant.shards.router import normalize_token

    return normalize_token(word) in _UNIT_TO_FAMILY


def _read_structure(text: str, allow_words: bool):
    """Split into operators, numbers-with-units and word runs.

    Structure only: nothing is looked up here. That order matters
    more than it looks. Resolving a word run before knowing whether
    the text even contains an operator would let this branch refuse
    aloud - "I hold nothing for 'north gate colour'" - for every
    ordinary question about something unheld. The operator check
    comes first, so only genuine arithmetic ever reaches a lookup.
    """
    from ultraquant.shards.router import normalize_token

    raw = _TOKEN_RE.findall(text)
    if not raw:
        return None
    if "".join(raw) != re.sub(r"\s+", "", text):
        # Something was not tokenizable (a stray symbol, a word with
        # digits in it) - not our question.
        return None
    if allow_words and re.search(r"[a-z]-[a-z]", text):
        # A hyphenated word is a word, not a subtraction: "t-shirt"
        # must never read as t minus shirt.
        return None
    items: list = []
    words: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if _NUMBER_RE.fullmatch(token):
            if words:
                items.append(("words", words))
                words = []
            unit = ""
            if (allow_words and index + 1 < len(raw)
                    and _is_unit(raw[index + 1])):
                unit = normalize_token(raw[index + 1])
                index += 1
            items.append(("number", token, unit))
        elif token in "()+-*/":
            if words:
                items.append(("words", words))
                words = []
            items.append(("op", token))
        elif token in _WORD_OPS:
            if words:
                items.append(("words", words))
                words = []
            items.append(("op", _WORD_OPS[token]))
            if (token in ("divided", "multiplied")
                    and index + 1 < len(raw) and raw[index + 1] == "by"):
                index += 1
        elif allow_words:
            words.append(token)
        else:
            return None
        index += 1
    if words:
        items.append(("words", words))
    operators = [item for item in items
                 if item[0] == "op" and item[1] in "+-*/"]
    operands = [item for item in items
                if item[0] in ("number", "words")]
    if not operators or not operands:
        return None
    return items


def _resolve(words: list[str], memory, premises: list,
             confidences: list):
    """A word run as a held quantity, or a refusal that says why."""
    from ultraquant.reason.inference import _unit

    key = " ".join(words)
    for article in ("the ", "a ", "an "):
        if key.startswith(article):
            key = key[len(article):]
            break
    if not key:
        # A bare article is not an operand. "what is the plus size?"
        # reads "plus" as an operator and leaves "the" behind, and
        # refusing there would eat an ordinary question.
        raise _NotArithmetic
    record = memory.recall_fact(key)
    if record is None:
        raise Undefined(f"I hold nothing for '{key}'")
    if record.get("negated"):
        raise Undefined(
            f"I hold only a denial for '{key}' ({key} is not "
            f"{record.get('value', '')}), and a denial holds no "
            "number")
    shown = str(record.get("value", ""))
    match = _SIGNED_RE.search(shown)
    if match is None:
        raise Undefined(f"'{key}' holds no number ({shown})")
    premises.append((key, shown))
    confidences.append(float(record.get("confidence", 0.0)))
    number = (Fraction(match.group()) if _EXACT_RATIONALS
              else float(match.group()))
    return Quantity(number, _unit(shown))


def _tokenize(text: str, memory=None):
    """The parser's token list, plus premises and confidence.

    Returns ``(tokens, premises, confidence)`` or None. Word runs are
    resolved against ``memory`` when one is given; without it, any
    word at all means "not my question", which is exactly the
    strictness §11.76 shipped with.
    """
    items = _read_structure(text, allow_words=memory is not None)
    if items is None:
        return None
    tokens: list = []
    echoes: list[str] = []
    premises: list = []
    confidences: list = []
    resolved = 0
    failure: Undefined | None = None
    for item in items:
        if item[0] == "op":
            tokens.append(item[1])
            echoes.append(item[1])
        elif item[0] == "number":
            number = (Fraction(item[1]) if _EXACT_RATIONALS
                      else float(item[1]))
            quantity = Quantity(number, item[2])
            tokens.append(quantity)
            echoes.append(_show(quantity))
            resolved += 1
        else:
            before = len(premises)
            try:
                tokens.append(_resolve(item[1], memory, premises,
                                       confidences))
            except Undefined as refusal:
                if failure is None:
                    failure = refusal
                tokens.append(None)
                echoes.append(" ".join(item[1]))
                continue
            resolved += 1
            # A quantity that came from a belief is echoed by its
            # KEY, not its value, so the reading stays checkable:
            # "tower height * 3" says which fact was used, and the
            # premise line below says what it held.
            echoes.append(premises[before][0])
    if failure is not None:
        # An operand that would not resolve refuses aloud ONLY when
        # something else in the expression did: "the north gate
        # height * 2" is arithmetic with a gap and says so, while a
        # sentence where nothing resolves is not arithmetic at all
        # and belongs to whatever branch can read it.
        if resolved:
            raise failure
        raise _NotArithmetic
    confidence = min(confidences) if confidences else None
    return tokens, echoes, premises, confidence


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

    def expr(self) -> Quantity:
        value = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            value = _add(value, self.term(), 1 if op == "+" else -1)
        return value

    def term(self) -> Quantity:
        value = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take()
            right = self.unary()
            value = _mul(value, right) if op == "*" else _div(value,
                                                              right)
        return value

    def unary(self) -> Quantity:
        if self.peek() in ("+", "-"):
            op = self.take()
            value = self.unary()
            return (Quantity(-value.value, value.unit) if op == "-"
                    else value)
        return self.atom()

    def atom(self) -> Quantity:
        token = self.peek()
        if token is None:
            raise SyntaxError("expression ends early")
        if isinstance(token, Quantity):
            self.take()
            return token
        if token == "(":
            self.take()
            value = self.expr()
            if self.peek() != ")":
                raise SyntaxError("unclosed parenthesis")
            self.take()
            return value
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


def evaluate(text: str, memory=None) -> MathResult | None:
    """Evaluate an arithmetic question, or None if it is not one.

    None means "not my question" and is the common case; a MathResult
    with a refusal means "an arithmetic question with no answer",
    which is a different thing and is said aloud.

    With ``memory``, operands may also be units ("300 meters") and
    held beliefs ("the tower height"), and the result carries the
    premises it rests on. Without it, an expression is numbers and
    operators or it is not this module's question.
    """
    stripped = strip_question(text)
    if not stripped:
        return None
    try:
        read = _tokenize(stripped, memory)
    except _NotArithmetic:
        return None
    except Undefined as refusal:
        return MathResult(refusal=str(refusal))
    if read is None:
        return None
    tokens, echoes, premises, confidence = read
    if not any(tok in "+-*/" for tok in tokens
               if isinstance(tok, str)):
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
            expression=_echo(echoes), premises=premises,
            confidence=confidence)
    except Undefined as refusal:
        return MathResult(refusal=str(refusal),
                          expression=_echo(echoes),
                          premises=premises, confidence=confidence)
    except (SyntaxError, IndexError):
        return None
    if reader.at != len(reader.tokens):
        return None
    shown = _show(value)
    return MathResult(shown=shown, expression=_echo(echoes),
                      fractional="/" in render_exact(value.value),
                      premises=premises, confidence=confidence)
