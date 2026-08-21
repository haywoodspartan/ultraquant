# The native cognitive tier

C++17, no external dependencies, built with MSVC or any C++17
compiler. This is **not** the accelerator tier — `native/uq_core.cpp`
and friends speed up numeric kernels. This is a native build of the
cognitive core itself: exact arithmetic, the fact store, the thought
pipeline.

## The rule it lives under

The pure-Python tier defines the semantics. This tier reproduces
them, and "reproduces" is checked as **strings**: it is right when it
says the same thing Python says, not when it says something
defensible. Every piece is gated against Python as the oracle, in
`ultraquant/experiments/native*_gate.py`.

Nothing here is required at runtime. Without these binaries the
system runs entirely in Python, the parity pins skip, and the suite
is green — that is what it means for one tier to define the
semantics. What must never exist is a native tier that is present and
wrong.

## Build

```
powershell -File native/uq/build.ps1
```

Output goes to `native/uq/_build/`, which is not tracked: unlike the
accelerator DLLs, nothing degrades without it.

## What is here

| piece | source | gate |
|---|---|---|
| arbitrary-precision integers | `src/bigint.cpp` | `nativebigint_gate` |
| exact rationals | `src/rational.cpp` | (through the reader) |
| the arithmetic reader | `src/calculate.cpp` | `nativecalc_gate` |
| the fact store | `src/memory.cpp` | `nativememory_gate` |
| the thought pipeline (spine) | `src/interpreter.cpp` | `nativechat_gate` |
| spreading-activation inference | `src/inference.cpp` | `nativeinfer_gate` |
| superlatives and aggregates | `src/forms.cpp` | `nativeforms_gate` |

Arbitrary precision comes first because C++ has none and the system
leans on Python's everywhere it matters — `2 ^ 1000` is a 302-digit
answer, an exact rational's denominator grows without being asked,
and a root enclosure scales its radicand by `10^(places*degree)`
before taking an integer root. A tier built on `long long` would
diverge on the first interesting question.

Two behaviours are implemented explicitly rather than inherited from
C++, because that is where a port disagrees silently: **floor
division** (Python rounds toward negative infinity and the remainder
takes the divisor's sign; C++ truncates toward zero) and **integer
roots** (bisection over integers, no float anywhere near a value).

## Running it

```
native/uq/_build/uq_chat.exe --plain
```

Type statements and questions; it learns and answers. What it does
NOT yet cover is listed in `nativechat_gate`'s docstring rather than
left to be discovered.

## A note on regexes

Every regex in this tree is a **raw string literal**. Two ordinary
literals in the list reader once lost their doubled backslashes in
transit and compiled to a pattern matching the letter "s"; the list
path went silently dead and MSVC's `warning C4129` was the only
witness. Raw literals mean the escaping question cannot be asked
twice.
