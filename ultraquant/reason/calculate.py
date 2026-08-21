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

**Percentages are structural** (§11.80): a percentage is a number
over a hundred and "of" is the multiplication it waits for, so
nothing downstream needs a special case - and "300 + 20%" refuses,
because taking the percentage of the left operand is a convention
rather than an arithmetic.

**Powers are exact and roots are honest** (§11.81). A whole exponent
is exact, the sign sits outside the power (-2^2 is -4), and the
exponent binds tighter than multiplication. A root that is rational
is given exactly; a root that is not is ENCLOSED - "sqrt 2 is
between 1.414213562 and 1.414213563" - with both bounds proved by
integer arithmetic and verified as rationals, because
1.4142135623730951 is not the square root of two.

**Rounding is a request** (§11.84). Exactness is the default; "to 2
decimal places" is parsed off the tail of the question, the
arithmetic still runs exactly, and only the answer is rounded - at
exactly the width that was asked for, saying that it was rounded and
naming the exact value it came from. A half rounds away from zero,
in exact rational arithmetic, so the tie rule is a decision rather
than an artefact. A request this module cannot serve (significant
figures) is refused by name rather than read as part of the
expression.

Two lines hold, in the house's tradition. Division by zero REFUSES
aloud - it is undefined, and a calculator that returns something for
it is lying about arithmetic itself. And the reader would rather
decline than guess: without a memory, any token that is not a
number, an operator or a parenthesis means "not my question"; with
one, an expression where NOTHING resolves means the same, so "what
is the plus size?" - which reads "plus" as an operator - falls
through untouched; and a polar lead ("is 2 + 2 equal to 4?") belongs
to §11.83, not here. A math branch that swallowed "what is the tower
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


class _Irrational(Exception):
    """A root with no exact value, carrying what to bound."""

    def __init__(self, value, degree: int) -> None:
        super().__init__("irrational")
        self.value = value
        self.degree = degree


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


def _iroot(number: int, degree: int) -> int:
    """The integer part of ``number ** (1/degree)``, exactly.

    Integer arithmetic throughout - the point of this module is that
    a float never touches a value, and a root is exactly where a
    float would be most tempting and least honest.
    """
    if number < 0:
        raise ValueError("negative radicand")
    if number < 2:
        return number
    if degree == 2:
        import math

        return math.isqrt(number)
    low, high = 0, 1
    while high ** degree <= number:
        high *= 2
    while low < high:
        middle = (low + high + 1) // 2
        if middle ** degree <= number:
            low = middle
        else:
            high = middle - 1
    return low


def _exact_root(value: Fraction, degree: int) -> Fraction | None:
    """The root as a rational, or None if it is irrational."""
    negative = value < 0
    if negative and degree % 2 == 0:
        raise Undefined(
            f"no real number raised to the power {degree} gives "
            f"{render_exact(value)}")
    top = _iroot(abs(value.numerator), degree)
    bottom = _iroot(value.denominator, degree)
    if (top ** degree != abs(value.numerator)
            or bottom ** degree != value.denominator):
        return None
    root = Fraction(top, bottom)
    return -root if negative else root


def _enclose(value: Fraction, degree: int, places: int = 9):
    """Exact decimal bounds for an irrational root.

    A root that has no exact value still has an exact PLACE: it lies
    between two decimals that can be proved, by integer arithmetic,
    to sit on either side of it. Naming that interval is honest in a
    way that naming 1.4142135623730951 is not - that number is not
    the square root of two, it is merely near it, and a system that
    says "absence is never no" about beliefs should not quietly
    round about numbers.
    """
    scale = 10 ** places
    low = Fraction(_iroot(abs(value.numerator) * scale ** degree
                          * value.denominator ** (degree - 1),
                          degree),
                   scale * value.denominator)
    # Prove the bracket rather than trusting the scaling: the floor
    # above can land a step low, and an unproved bound is a guess.
    while low ** degree > value:
        low -= Fraction(1, scale)
    while (low + Fraction(1, scale)) ** degree <= value:
        low += Fraction(1, scale)
    return low, low + Fraction(1, scale)


def _power(left: Quantity, right: Quantity) -> Quantity:
    """``left ** right`` where the exponent is a whole number."""
    if right.unit:
        raise Undefined(
            f"an exponent in {right.unit}s is not a number of times")
    exponent = right.value
    if isinstance(exponent, Fraction):
        if exponent.denominator != 1:
            raise Undefined(
                "an exponent that is not a whole number asks for a "
                "root - ask me for the root itself and I will bound "
                "it exactly")
        exponent = exponent.numerator
    elif exponent != int(exponent):
        raise Undefined(
            "an exponent that is not a whole number asks for a root "
            "- ask me for the root itself and I will bound it "
            "exactly")
    else:
        exponent = int(exponent)
    if abs(exponent) > _MAX_EXPONENT:
        raise Undefined(
            f"an exponent past {_MAX_EXPONENT} names a number with "
            "more digits than an answer has")
    if left.value == 0 and exponent == 0:
        raise Undefined(
            "zero to the power zero is a convention, not a value - "
            "different fields choose differently, and I will not "
            "choose for you")
    if left.value == 0 and exponent < 0:
        raise ZeroDivisionError
    if left.unit and exponent != 1:
        raise Undefined(
            f"raising {left.unit}s to a power would name a unit no "
            "definition I hold covers")
    return Quantity(left.value ** exponent, left.unit)


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
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[()+\-*/%^]|[a-z']+")

#: Root words, mapped to their degree. Prefix functions in the
#: grammar, so "sqrt 16" and "sqrt(16)" are the same reading.
_ROOT_DEGREES = {"sqrt": 2, "cbrt": 3}

#: Spoken forms rewritten to the symbols the grammar already has, so
#: "5 squared" and "5 ^ 2" are one reading and not two code paths.
_SPOKEN_POWERS = (
    (re.compile(r"\b(?:the\s+)?square\s+roots?\s+of\b"), "sqrt"),
    (re.compile(r"\b(?:the\s+)?(?:cube|cubic)\s+roots?\s+of\b"),
     "cbrt"),
    (re.compile(r"\bsquare\s+roots?\b"), "sqrt"),
    (re.compile(r"\b(?:cube|cubic)\s+roots?\b"), "cbrt"),
    (re.compile(r"\bsqrt\s+of\b"), "sqrt"),
    (re.compile(r"\bcbrt\s+of\b"), "cbrt"),
    (re.compile(r"\bto\s+the\s+power\s+of\b"), "^"),
    (re.compile(r"\braised\s+to\b"), "^"),
    (re.compile(r"\bsquared\b"), "^ 2"),
    (re.compile(r"\bcubed\b"), "^ 3"),
)

#: A rounding REQUEST, parsed off the tail of a question. Exactness
#: is the default; rounding is something a speaker asks for, and an
#: answer that was rounded says so.
_WORD_PLACES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10}
_ROUNDING_RES = (
    (re.compile(r"[, ]*(?:rounded\s+)?to\s+(\d+|one|two|three|four|"
                r"five|six|seven|eight|nine|ten)\s+"
                r"(?:decimal\s+)?places?\s*$"), None),
    (re.compile(r"[, ]*(?:rounded\s+)?to\s+the\s+nearest\s+"
                r"(?:whole\s+number|integer)\s*$"), 0),
)

#: Rounding requests this module does NOT serve. Named rather than
#: ignored: a request it cannot honour should be refused by name,
#: not read as part of the expression and refused about the word
#: "to".
_UNSUPPORTED_ROUNDING = re.compile(
    r"[, ]*(?:rounded\s+)?to\s+(?:\d+|one|two|three|four|five|six|"
    r"seven|eight|nine|ten)\s+(?:significant\s+figures?|"
    r"sig\s*figs?)\s*$"
    r"|[, ]*(?:rounded\s+)?to\s+the\s+nearest\s+"
    r"(?:ten|hundred|thousand|tenth|hundredth)\s*$")

#: The §11.87 rung: a list of WRITTEN numbers ("the average of 3, 5
#: and 10"), which is a different question from §11.57's aggregate
#: over held facts even though the words are the same. Every item
#: must be written, so a list naming beliefs falls through to the
#: machinery that owns beliefs. The list gate's baseline arm turns
#: it off.
_LISTS_ON = True

#: List words, folded to the four operations they name.
_LIST_WORDS = {
    "sum": "sum", "total": "sum",
    "average": "average", "mean": "average",
    "largest": "largest", "biggest": "largest",
    "maximum": "largest", "max": "largest", "greatest": "largest",
    "smallest": "smallest", "minimum": "smallest", "min": "smallest",
    "least": "smallest",
}

_LIST_RE = re.compile(
    r"^(?:the\s+)?(" + "|".join(sorted(_LIST_WORDS, key=len,
                                        reverse=True))
    + r")\s+of\s+(.+)$")

#: Past this, the answer has more digits than an answer has.
_MAX_EXPONENT = 1000

#: When False the module evaluates in binary floating point - the
#: standard option, implemented honestly, and the gate's baseline arm.
_EXACT_RATIONALS = True

#: The §11.84 rung: exactness is the default and rounding is a
#: REQUEST. "to 2 decimal places" is parsed off the tail of the
#: question, the arithmetic still runs exactly, and only the answer
#: is rounded - which it says, naming the exact value it came from.
#: The rounding gate's baseline arm turns it off.
_ROUNDING_ON = True

#: The §11.81 rung: powers and roots. Whole exponents are exact;
#: a root that IS rational is given exactly; a root that is not is
#: given as a proved decimal enclosure rather than a point value,
#: because 1.4142135623730951 is not the square root of two.
_POWERS_ON = True

#: The §11.80 rung: percentages. "20% of 300" is a hundredth part
#: taken 20 times and then multiplied, both halves structural; a
#: percentage with nothing to be a percentage OF refuses, because
#: "300 + 20%" is a convention rather than an arithmetic.
_PERCENTS_ON = True

#: The §11.79 rung: an operand the store does not HOLD may still be
#: REACHED, through the same chain machinery §11.55 uses for a
#: comparative operand, under the same three guards. The
#: derived-operand gate's baseline arm turns it off, leaving §11.78's
#: held-outright rule in place.
_DERIVE_OPERANDS = True


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
    bounds: tuple | None = None
    rounded_to: int | None = None
    was_rounded: bool = False
    exact_shown: str = ""

    @property
    def answered(self) -> bool:
        return not self.refusal and (self.shown != "" 
                                     or self.bounds is not None)


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
    if _POWERS_ON:
        for pattern, symbol in _SPOKEN_POWERS:
            low = pattern.sub(symbol, low)
    return low.strip()


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
    dangling_percent = False
    index = 0
    while index < len(raw):
        token = raw[index]
        if _NUMBER_RE.fullmatch(token):
            if words:
                items.append(("words", words))
                words = []
            following = raw[index + 1] if index + 1 < len(raw) else ""
            if _PERCENTS_ON and following in ("%", "percent"):
                # §11.80: a percentage is a number divided by a
                # hundred, and "of" is the multiplication it is
                # waiting for. Both halves are structural: nothing
                # here is a special case for percents downstream.
                items.append(("percent", token))
                index += 1
                if (index + 1 < len(raw)
                        and raw[index + 1] in ("of", "off")):
                    items.append(("op", "*", "of"))
                    index += 1
                else:
                    dangling_percent = True
                index += 1
                continue
            unit = ""
            if (allow_words and index + 1 < len(raw)
                    and _is_unit(raw[index + 1])):
                unit = normalize_token(raw[index + 1])
                index += 1
            items.append(("number", token, unit))
        elif token == "%":
            # Only reachable when the percent branch above did not
            # take it - a sign with no number in front of it is not
            # arithmetic, and with the rung off it never is.
            return None
        elif token in _ROOT_DEGREES and _POWERS_ON:
            if words:
                items.append(("words", words))
                words = []
            items.append(("root", token))
        elif token == "^":
            if not _POWERS_ON:
                return None
            if words:
                items.append(("words", words))
                words = []
            items.append(("op", "^"))
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
                 if (item[0] == "op" and item[1] in "+-*/^")
                 or item[0] == "root"]
    operands = [item for item in items
                if item[0] in ("number", "words", "percent")]
    if not operators or not operands:
        return None
    if dangling_percent:
        # "300 + 20%" is a convention, not an arithmetic: twenty per
        # cent OF WHAT is a question the sentence does not answer,
        # and picking the left operand for the speaker would be
        # guessing at their meaning in the one branch that exists to
        # be exact.
        raise Undefined(
            "a percentage needs something to be a percentage of - "
            "'20% of 300' has an answer, '300 + 20%' has a "
            "convention")
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
    trail = ""
    if record is None and _DERIVE_OPERANDS:
        # §11.79: the store may not HOLD the operand and still reach
        # it. §11.55 already derives a comparative operand this way,
        # and the two must not disagree about what a store knows:
        # "is the west tower taller than the south tower?" reaching
        # through `west tower kind is spirekind` while "the west
        # tower height * 2" refuses would be one matrix with two
        # voices. The same three guards ride along - no modifier-
        # rescued subject, no derived denial, no empty conclusion.
        from ultraquant.reason.inference import infer

        attempt = infer(f"what is the {key}?", memory)
        if (attempt is not None and attempt.conclusion is not None
                and "as a modifier" not in attempt.answer
                and "as modifiers" not in attempt.answer):
            if attempt.negated:
                raise Undefined(
                    f"I can only derive a denial for '{key}' ({key} "
                    f"is not {attempt.conclusion[1]}), and a denial "
                    "holds no number")
            bridges = ", ".join(str(value)
                                for _k, value in attempt.premises[:-1])
            trail = f" (derived via {bridges})" if bridges else \
                    " (derived)"
            record = {"value": attempt.conclusion[1],
                      "confidence": attempt.confidence}
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
    premises.append((key, shown + trail))
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
            echoes.append(item[2] if len(item) > 2 else item[1])
        elif item[0] == "root":
            tokens.append(item[1])
            echoes.append(item[1])
        elif item[0] == "percent":
            hundredths = (Fraction(item[1]) / 100 if _EXACT_RATIONALS
                          else float(item[1]) / 100)
            tokens.append(Quantity(hundredths, ""))
            echoes.append(f"{item[1]}%")
            resolved += 1
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
        return self.power()

    def power(self) -> Quantity:
        """``atom ^ unary``, right-associative - §11.81.

        The sign sits OUTSIDE the power, which is what "-2^2 = -4"
        means, and the right operand is a unary so "2^-3" reads. The
        exponent binds tighter than multiplication because that is
        what everyone writing "3 * 2^4" means.
        """
        value = self.atom()
        if self.peek() == "^":
            self.take()
            return _power(value, self.unary())
        return value

    def atom(self) -> Quantity:
        token = self.peek()
        if token is None:
            raise SyntaxError("expression ends early")
        if isinstance(token, Quantity):
            self.take()
            return token
        if token in _ROOT_DEGREES:
            self.take()
            degree = _ROOT_DEGREES[token]
            # A unary, not an atom: "sqrt -4" must reach the root to
            # be refused there, rather than dying as a syntax error
            # and falling through as though it were not arithmetic.
            inner = self.unary()
            if inner.unit:
                raise Undefined(
                    f"the root of a length in {inner.unit}s would "
                    "name a unit no definition I hold covers")
            if not isinstance(inner.value, Fraction):
                # The float arm: the standard option, which answers
                # with a number that is not the root. Its domain
                # error is kept honest rather than allowed to return
                # a complex number, which is Python's answer to
                # (-4) ** 0.5 and nobody's answer to the question.
                if inner.value < 0:
                    if degree % 2 == 0:
                        raise Undefined(
                            f"no real number raised to the power "
                            f"{degree} gives "
                            f"{render_exact(inner.value)}")
                    return Quantity(-((-inner.value)
                                      ** (1.0 / degree)), "")
                return Quantity(inner.value ** (1.0 / degree), "")
            exact = _exact_root(inner.value, degree)
            if exact is not None:
                return Quantity(exact, "")
            raise _Irrational(inner.value, degree)
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
    return (index == 0 or tokens[index - 1] in "(+-*/^"
            or tokens[index - 1] in _ROOT_DEGREES)


def _fixed(value: Fraction, places: int) -> str:
    """A rational printed at exactly ``places`` decimals.

    A request for two decimal places is answered in two decimal
    places: "10.00", not "10". The exact renderer prints the
    shortest true form, which is right when nobody named a
    precision and wrong when somebody did - the padding is part of
    the answer, because it says how far the claim goes.
    """
    if places == 0:
        return str(value.numerator // value.denominator)
    scaled = value * (10 ** places)
    whole = scaled.numerator // scaled.denominator
    digits = str(abs(whole)).rjust(places + 1, "0")
    sign = "-" if value < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def _round_to(value: Fraction, places: int) -> Fraction:
    """``value`` rounded to ``places`` decimals, half away from zero.

    Exact rational arithmetic, so the tie rule is a decision rather
    than an artefact of how a float happened to land: a half rounds
    away from zero, which is the rule people are taught and expect.
    """
    scale = 10 ** places
    shifted = value * scale
    floor = shifted.numerator // shifted.denominator
    remainder = shifted - floor
    if remainder > Fraction(1, 2):
        floor += 1
    elif remainder == Fraction(1, 2):
        floor += 1 if value >= 0 else 0
    return Fraction(floor, scale)


def _rounded_root(value: Fraction, degree: int, places: int):
    """A root rounded to ``places``, proved rather than approximated.

    The root is enclosed at increasing precision until BOTH bounds
    round to the same decimal; then that decimal is the correct
    rounding, and it is the correct rounding because two proved
    bounds agree on it, not because a float looked like it.
    """
    extra = places + 3
    while True:
        low, high = _enclose(value, degree, extra)
        down, up = _round_to(low, places), _round_to(high, places)
        if down == up:
            return down
        extra += 3


def read_list(text: str) -> MathResult | None:
    """Answer "what is the average of 3, 5 and 10?" - §11.87.

    A list of WRITTEN numbers, which is a different question from
    §11.57's aggregate over held facts even though the words are the
    same. The boundary is deliberate and it is where the two voices
    are kept apart: every item must parse as a written number or
    quantity, so a list naming beliefs falls through untouched to the
    machinery that owns beliefs. One store, one voice per question.

    Units ride the list the way they ride the operators: same family
    converts and reads in the LARGEST unit present, a mixed family
    refuses, and a bare number beside a united one refuses, because
    assuming its unit would be invention.
    """
    if not _LISTS_ON:
        return None
    stripped = strip_question(text)
    match = _LIST_RE.match(stripped)
    if match is None:
        return None
    operation = _LIST_WORDS[match.group(1)]
    body = match.group(2).strip().rstrip("?.")
    pieces = [piece.strip()
              for piece in re.split(r",|\band\b", body)
              if piece.strip()]
    if len(pieces) < 2:
        # "the average of 5" is not a list, and answering it would be
        # a parlour trick rather than arithmetic.
        return None
    items = []
    for piece in pieces:
        quantity = read_quantity(piece)
        if quantity is None:
            return None
        items.append(quantity)
    try:
        total = items[0]
        for item in items[1:]:
            total = _add(total, item, 1)
        if operation in ("largest", "smallest"):
            aligned = []
            for item in items:
                left, _right, unit = _aligned(item, items[0])
                aligned.append((left, unit))
            pick = (max if operation == "largest" else min)(
                aligned, key=lambda pair: pair[0])
            value = Quantity(pick[0], pick[1])
        elif operation == "average":
            value = Quantity(total.value / len(items), total.unit)
        else:
            value = total
    except Undefined as refusal:
        return MathResult(refusal=str(refusal))
    except ZeroDivisionError:                # pragma: no cover
        return None
    spoken = ", ".join(_show(item) for item in items[:-1])
    spoken = f"{spoken} and {_show(items[-1])}"
    return MathResult(shown=_show(value),
                      expression=f"the {operation} of {spoken}",
                      fractional="/" in render_exact(value.value))


def read_quantity(text: str) -> Quantity | None:
    """A bare written operand - "200" or "200 meters" - or None.

    §11.82 needs this so a comparison can have a written right-hand
    side: "is the tower height greater than 200 meters?" compares a
    belief with a quantity nobody stored. Deliberately narrow - a
    number, optionally followed by one known unit, and nothing else -
    because anything looser would start reading keys, and keys are
    the caller's business.
    """
    tokens = _TOKEN_RE.findall(text.strip().lower())
    if not tokens or not _NUMBER_RE.fullmatch(tokens[0]):
        return None
    if len(tokens) == 1:
        return Quantity(Fraction(tokens[0]), "")
    if len(tokens) == 2 and _is_unit(tokens[1]):
        from ultraquant.shards.router import normalize_token

        return Quantity(Fraction(tokens[0]),
                        normalize_token(tokens[1]))
    return None


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
    places = None
    if _ROUNDING_ON:
        for pattern, fixed in _ROUNDING_RES:
            match = pattern.search(stripped)
            if match is not None:
                places = (fixed if fixed is not None
                          else _WORD_PLACES.get(match.group(1))
                          or int(match.group(1)))
                stripped = stripped[:match.start()].strip()
                break
    if places is None and _ROUNDING_ON:
        if _UNSUPPORTED_ROUNDING.search(stripped):
            return MathResult(
                refusal="I can round to a number of decimal places "
                        "or to the nearest whole number; that is a "
                        "different rule and I do not have it")
    if stripped.startswith(("is ", "are ", "was ", "were ")):
        # A polar lead asks whether something HOLDS, which is
        # §11.83's question and not this one. Reading it here made
        # "is 2 + 2 equal to 4?" refuse with "I hold nothing for
        # 'is'" - a refusal about the wrong word entirely.
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
    if not any(tok in "+-*/^" or tok in _ROOT_DEGREES
               for tok in tokens if isinstance(tok, str)):
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
    except _Irrational as irrational:
        if places is not None:
            # A precision was ASKED for, so the enclosure collapses
            # to the decimal both of its bounds agree on.
            value = _rounded_root(irrational.value,
                                  irrational.degree, places)
            return MathResult(shown=_fixed(value, places),
                              expression=_echo(echoes),
                              rounded_to=places, was_rounded=True,
                              exact_shown="irrational",
                              premises=premises,
                              confidence=confidence)
        low, high = _enclose(irrational.value, irrational.degree)
        return MathResult(expression=_echo(echoes),
                          bounds=(render_exact(low),
                                  render_exact(high)),
                          premises=premises, confidence=confidence)
    except Undefined as refusal:
        return MathResult(refusal=str(refusal),
                          expression=_echo(echoes),
                          premises=premises, confidence=confidence)
    except (SyntaxError, IndexError):
        return None
    if reader.at != len(reader.tokens):
        return None
    if places is not None and isinstance(value.value, Fraction):
        exact = value.value
        value = Quantity(_round_to(exact, places), value.unit)
        padded = _fixed(value.value, places)
        shown = (f"{padded} {value.unit}s" if value.unit else padded)
        return MathResult(shown=shown,
                          expression=_echo(echoes),
                          rounded_to=places,
                          was_rounded=value.value != exact,
                          exact_shown=render_exact(exact),
                          premises=premises, confidence=confidence)
    shown = _show(value)
    return MathResult(shown=shown, expression=_echo(echoes),
                      fractional="/" in render_exact(value.value),
                      premises=premises, confidence=confidence)
