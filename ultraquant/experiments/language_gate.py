"""Stage 5's gate: correct sentences it was never given a template for.

[ARCHITECTURE.md §11.3] states it before the work:

    Templates replaced by composition over learned concepts. Gate: produces
    correct sentences it was never given a template for.

The hollow version is a template with slots — still a template, however the
slots are filled. So the hidden languages live *here*, in the experiment, and
the learner sees only ``(utterance, meaning)`` pairs. A test asserts the
languages' function words appear nowhere in ``ultraquant/reason/language.py``.

Pre-registered clauses, all required:

1. **Held-out combinations** (depths 2-3, parts seen, pairings never):
   exact match >= 0.80 and content-word match >= 0.90 against the hidden
   grammar's own output.
2. **A second, structurally different language** (mark last, containers
   outer-first, different function words) learned by the same code to the same
   bar. A baked-in template can satisfy one language; it cannot satisfy both.
3. **Depth extrapolation**: trained on depths 2-3 only, exact >= 0.80 at
   depth 4 — utterances one recursion deeper than anything ever seen.
4. **Comprehension**: held-out utterances parsed back to their meanings,
   >= 0.80 exact.
5. **Control**: a language with no consistent word-meaning mapping must score
   <= 0.10 exact, or the measurement itself is broken.

A memoriser baseline (training pairs in a dict) scores exactly 0 on held-out
combinations by construction; it is reported as the sanity line, not as the
competitor — that lesson is from Stage 2.

Run it::

    python -m ultraquant.experiments.language_gate
"""

from __future__ import annotations

import random
import statistics
import sys

from ultraquant.reason.language import GroundedLanguage

__all__ = ["GRAMMARS", "hidden_utterance", "meanings", "one_trial", "run_gate", "main"]

#: Pre-registered thresholds.
EXACT_TARGET = 0.80
CONTENT_TARGET = 0.90
CONTROL_CEILING = 0.10

#: Internal atom names deliberately differ from several surface words, so
#: string matching cannot substitute for learned alignment.
MARK_WORDS = {"plus": "plus", "ex": "saltire", "dot": "speck", "bar": "dash"}
SHAPE_WORDS = {"box": "box", "rails": "grill", "caps": "caps", "corners": "nooks"}

#: The two hidden languages. Everything here is invisible to the learner's code.
GRAMMARS = {
    "nested": {  # "a speck inside a box inside a grill" — mark first, inner-first
        "mark_first": True, "inner_first": True,
        "prefix": ("a",), "gap": ("inside", "a"), "suffix": (),
    },
    "holding": {  # "a grill holding a box holding a speck" — outer-first, mark last
        "mark_first": False, "inner_first": False,
        "prefix": ("a",), "gap": ("holding", "a"), "suffix": (),
    },
}


def hidden_utterance(levels: list[str], grammar: dict) -> str:
    """The reference sentence for a meaning, from the hidden grammar."""
    containers = [SHAPE_WORDS[value] for value in levels[:-1]]
    mark = MARK_WORDS[levels[-1]]
    if grammar["inner_first"]:
        containers = list(reversed(containers))
    words = [mark] + containers if grammar["mark_first"] else containers + [mark]
    tokens = list(grammar["prefix"])
    for index, word in enumerate(words):
        if index:
            tokens.extend(grammar["gap"])
        tokens.append(word)
    tokens.extend(grammar["suffix"])
    return " ".join(tokens)


def meanings(depth: int) -> list[list[str]]:
    """Every meaning at ``depth`` levels (distinct containers, one mark)."""
    shapes = sorted(SHAPE_WORDS)
    marks = sorted(MARK_WORDS)
    if depth == 2:
        return [[shape, mark] for shape in shapes for mark in marks]
    out = []
    for outer in shapes:
        for inner in shapes:
            if inner == outer:
                continue
            if depth == 3:
                out.extend([[outer, inner, mark] for mark in marks])
            else:
                for third in shapes:
                    if third in (outer, inner):
                        continue
                    out.extend([[outer, inner, third, mark] for mark in marks])
    return out


def _split(rng: random.Random) -> tuple[list[list[str]], list[list[str]]]:
    """Training and held-out meanings over depths 2-3, every part seen."""
    depth2 = meanings(2)
    depth3 = meanings(3)
    for _attempt in range(300):
        rng.shuffle(depth2)
        rng.shuffle(depth3)
        held = depth2[:6] + depth3[:10]
        train = depth2[6:] + depth3[10:]
        seen_shapes = {value for levels in train for value in levels[:-1]}
        seen_marks = {levels[-1] for levels in train}
        if seen_shapes == set(SHAPE_WORDS) and seen_marks == set(MARK_WORDS):
            return train, held
    raise RuntimeError("could not split while keeping every part in training")


def _content(utterance: str) -> tuple:
    """The multiset of content words, for the weaker match."""
    content_words = set(MARK_WORDS.values()) | set(SHAPE_WORDS.values())
    return tuple(sorted(token for token in utterance.split()
                        if token in content_words))


def one_trial(seed: int, grammar_name: str = "nested") -> dict:
    """Train on pairs from one hidden grammar; score every clause."""
    rng = random.Random(seed)
    grammar = GRAMMARS[grammar_name]
    train, held = _split(rng)

    model = GroundedLanguage()
    fit = model.fit([(hidden_utterance(levels, grammar), levels)
                     for levels in train])

    def score(test_meanings: list[list[str]]) -> tuple[float, float]:
        exact = 0
        content = 0
        for levels in test_meanings:
            reference = hidden_utterance(levels, grammar)
            produced = model.generate(levels)
            exact += produced == reference
            content += (produced is not None
                        and _content(produced) == _content(reference))
        n = max(1, len(test_meanings))
        return exact / n, content / n

    held_exact, held_content = score(held)

    depth4 = meanings(4)
    rng.shuffle(depth4)
    deep_exact, _deep_content = score(depth4[:24])

    parsed = sum(
        model.parse(hidden_utterance(levels, grammar)) == levels
        for levels in held
    ) / max(1, len(held))

    memorised = {tuple(levels): hidden_utterance(levels, grammar)
                 for levels in train}
    memoriser = sum(tuple(levels) in memorised for levels in held) / max(1, len(held))

    # Control: same meanings, but each sentence is random words — no
    # consistent mapping exists, so an honest learner has nothing to find.
    vocabulary = sorted(set(MARK_WORDS.values()) | set(SHAPE_WORDS.values())
                        | set(grammar["prefix"]) | set(grammar["gap"]))
    control = GroundedLanguage()
    control.fit([
        (" ".join(rng.choice(vocabulary)
                  for _ in range(len(hidden_utterance(levels, grammar).split()))),
         levels)
        for levels in train
    ])
    control_exact = sum(
        control.generate(levels) == hidden_utterance(levels, grammar)
        for levels in held
    ) / max(1, len(held))

    return {
        "coverage": fit["coverage"],
        "held_exact": held_exact,
        "held_content": held_content,
        "depth4_exact": deep_exact,
        "parse": parsed,
        "memoriser": memoriser,
        "control_exact": control_exact,
    }


def run_gate(seeds: int = 10, grammar_name: str = "nested") -> dict:
    """The gate over ``seeds`` train/held splits."""
    trials = [one_trial(seed, grammar_name) for seed in range(seeds)]

    def mean(key: str) -> float:
        return statistics.fmean(t[key] for t in trials)

    result = {key: mean(key) for key in trials[0]}
    result["seeds"] = seeds
    result["grammar"] = grammar_name
    result["passes"] = (
        result["held_exact"] >= EXACT_TARGET
        and result["held_content"] >= CONTENT_TARGET
        and result["depth4_exact"] >= EXACT_TARGET
        and result["parse"] >= EXACT_TARGET
        and result["control_exact"] <= CONTROL_CEILING
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run every clause for both hidden languages and print a verdict."""
    argv = list(sys.argv[1:] if argv is None else argv)
    seeds = int(argv[0]) if argv else 10

    print("Stage 5 gate: correct sentences it was never given a template for")
    print(f"pre-registered: exact >= {EXACT_TARGET:.2f}, content >= "
          f"{CONTENT_TARGET:.2f}, depth-4 >= {EXACT_TARGET:.2f}, parse >= "
          f"{EXACT_TARGET:.2f}, control <= {CONTROL_CEILING:.2f}; {seeds} seeds")

    verdicts = []
    for name in GRAMMARS:
        result = run_gate(seeds=seeds, grammar_name=name)
        verdicts.append(result)
        print(f"\nlanguage '{name}'"
              f"  (example: \"{hidden_utterance(['rails', 'box', 'dot'], GRAMMARS[name])}\")")
        print(f"  lexicon coverage        {result['coverage']:.2f}")
        print(f"  held-out combos: exact  {result['held_exact']:.3f}   "
              f"content {result['held_content']:.3f}")
        print(f"  depth-4 (never seen)    {result['depth4_exact']:.3f}")
        print(f"  parse (comprehension)   {result['parse']:.3f}")
        print(f"  memoriser baseline      {result['memoriser']:.3f}  "
              f"(0 by construction)")
        print(f"  control (no mapping)    {result['control_exact']:.3f}")
        print(f"  clauses met             {result['passes']}")

    print("\nverdict")
    if all(v["passes"] for v in verdicts):
        print("  PASS - both hidden languages are learned by the same code, novel")
        print("  combinations and a novel depth are spoken exactly, comprehension")
        print("  inverts it, and the control confirms nothing was baked in.")
    else:
        failed = [v["grammar"] for v in verdicts if not v["passes"]]
        print(f"  FAIL - clauses unmet for: {', '.join(failed)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
