# UltraQuant

A **hybrid quantum/classical pattern library** built entirely from scratch in
pure Python stdlib — no numpy, no ML frameworks. The model is not a monolith:
it is a **sharded, catalogued library of learned patterns** that stays on disk
and pages in only what a thought needs, the way a brain recalls rather than
reloads. Optional C++/CUDA accelerators and real quantum hardware are strictly
accelerants — the pure-Python tier defines the semantics, and every other tier
reproduces it to 1e-9.

To be clear about scope up front: **this is not AGI, and it does not claim
quantum advantage.** What it is: a system with unusually honest machinery,
where every capability sits behind a pre-registered, measured gate — and the
failures are documented next to the passes
([ARCHITECTURE.md §11](ARCHITECTURE.md)).

## What it does

| capability | the measurement behind it |
|---|---|
| **Pageable shard library** | 1:6,200 resident-to-stored at the 1.2T-parameter design point; ships between machines inside `.uql` containers |
| **Three-tier storage split** | disk holds the library, DRAM holds the working set, VRAM holds the hot layers (weights cross the bus once: 0.061 -> 0.028 ms/query at 3.6e-15 parity on an RTX 4090) |
| **From-scratch quantum engine** | exact statevector sim, Mottonen encoding, parameter-shift gradients, kernels, ZNE, Grover; IBM QPU / BlueQubit optional |
| **Content routing** | sketch-screened cosine over stored prototypes: 47x faster than exact scan at 20k categories, exact rerank always |
| **Composition** (blackboard) | reads compound scenes a monolithic classifier scores 0.000 on (0.539, at 2.4x seed variance) |
| **Self-proposed concepts** | clusters what it could not place and asks *you* what to call the group (ARI 1.000, class count never supplied) |
| **Goal planning** | multi-step action sequences from goals alone — no per-task scripts; capabilities added later are usable without planner changes |
| **Grounded language** | induced lexicon, order and frame — no templates, asserted over the AST; speaks what it perceives, in three languages, at nesting depths never seen in training |
| **Self-learning loop** | finds its own gaps, asks the web first (through a found-is-not-believed quarantine), asks the human second |
| **Spreading-activation inference** | answers questions the library holds only implicitly: activation converges across stored facts; premises named, marked inferred, confidence diluted. Chains now span FIVE facts (four bridges) — each depth move gated on zero decoy assertions at the new depth — and synchronous frontier rounds run 3.2x faster at byte-identical parity |
| **Consolidation + truth maintenance** | say "yes" to a derivation and it becomes recallable memory usable as a premise (+0.625 reach at 1.21x sd); revise a premise and every conclusion resting on it retracts, recursively |
| **Curiosity from failed inference** | a question it cannot connect becomes a question it ASKS: the missing premise enters the learn queue at top rank, and answering it closes the fail->ask->learn->infer loop (0.750 closure at 1.62x sd, 1.000 premise precision, zero spam) |
| **Semantic reading at the boundary** | "how tall is the tower" reads as 'tower height', named and vetoable (synonym wall 0.000 -> 0.833 live at 2.04x sd, zero decoy falls); the embedding suggests, the lexical core verifies, and the chained half stays with the spread |
| **Evidence accumulation** | corroboration and contradiction typed and graded from live-web measurement; belief rises by degree, conflicts become questions |
| **Learned dispatch** | cores, threads and tiers decided by machine-learned experience — three brains (classical net / variational quantum circuit / one-qubit-per-core committee) in a measured shootout; 13x regret removed vs the static preference walk |
| **Polarity, end to end** | "is not steel" is belief-of-absence: stored as identity, inhibitory everywhere (denials never bridge chains, never join arithmetic, never win rankings), spoken aloud, and interrogable — yes against belief, no against contrary belief, honest don't-know otherwise; absence is never no (polar 0.273 -> 1.000 at 28.33x sd; the baseline derived through a denial 36 times, measured) |
| **Four tenses, one record** | what is (recall/derivation), what was (revision history, in order, polarity included), what happened (retractions name their premise), why (provenance: stated, consolidated, or derived) — zero rationalised false premises, zero invented histories or endings across every gate |
| **Comparatives to aggregates** | pairwise verdicts naming both values (operands derived through chains where unstored, each marked with its trail), setwise superlatives and count/total/average over the enumerated family — scope named, exclusions named, ties named, and a denial never moves a number (the pre-fix comparative coin fabricated 38 verdicts; the owner fabricated zero). The comparative word carries its attribute: "is A taller than B?" and "which is the tallest?" read height out of the word (0.000 -> 0.700 at 9.48x sd), while polysemous words — bigger, highest, even our own "shorter" — refuse to guess a family |
| **Lists of numbers** | "what is the average of 3, 5 and 10?" — sum, average, largest and smallest over WRITTEN numbers, with units converting and exactness inherited (the average of 1, 2 and 4 is 7/3). The rung is mostly a boundary: every item must be written out, so a list naming held facts falls through to the aggregate machinery that owns beliefs and keeps its voice (0.000 -> 0.804 at 11.27x sd, zero boundary answers moved) — the one-matrix-one-voice lesson applied before the bug rather than after it |
| **Rounding is a request** | exactness is the default and "to 2 decimal places" is something you ASK for: the arithmetic still runs exactly, only the answer is rounded, at the width requested, and it says so while naming the exact value it came from — "100 / 3 = 33.33, rounded to 2 decimal places; the exact value is 100/3" (0.000 -> 0.802 at 14.51x sd, zero unlabelled approximations). A rounded number that doesn't say it is rounded is the original failure wearing a shorter coat |
| **Asking whether the arithmetic holds** | "is 2 + 2 = 5?" and "is 3 * 4 greater than 10?" — either side of a comparison may be a number, a quantity, a belief or an expression, and equality pivots like any other comparison (0.341 -> 1.000 at 8.52x sd). The composition is the point: "is the tower height * 2 greater than 500 meters?" multiplies a belief keeping its unit, compares against a quantity nobody stored, and reads the left side as an expression — three rungs built separately meeting in one sentence, because each was built as a rule rather than a case |
| **A coin that always said No** | found by asking in the words people use: "is the tower height greater than 200 meters?" answered No for a 300-meter tower, and so did every other comparison in both directions — "greater", "more" and "less" were not in the comparative vocabulary, so the questions fell to the polar branch, which compared the claim to the stored value as STRINGS. The missing words joined; more importantly, a comparison the comparative branch declines is now refused by name ("I don't know how to compare by 'wider'") rather than answered (0.383 -> 1.000 at 3.23x sd). The baseline's 0.383 is not a score but the shape of the bug: always-No is right as often as the truth happens to be No, so it measures the worlds, not the system |
| **Powers, and roots that admit it** | whole powers exact (2^10, 2^-3, 0.1^2 = 0.01 not 0.010000000000000002), rational roots exact — and an irrational root given as PROVED BOUNDS rather than a value it does not have: "sqrt 2 is between 1.414213562 and 1.414213563", computed by integer arithmetic and verified as rationals. The standard option answers 1.4142135623730951, whose square is 2.0000000000000004 (0.702 -> 1.000 at 3.56x sd, zero false bounds — a wrong bracket would be a firmer lie than the float) |
| **Percentages, structurally** | "what is 20% of 300?" reads as a hundredth part taken and "of" as the multiplication it waits for, so percentages compose with units and chains without a special case anywhere downstream — "20% of the west tower height" is 180 meters, derived via spirekind (0.000 -> 0.832 at 17.28x sd). "300 + 20%" REFUSES: every calculator answers 360 by convention, and a convention is a guess at the speaker's meaning, which is the one thing an exact branch must not make |
| **Arithmetic over what is believed** | an operand may be a written number, a quantity ("300 meters"), or a held belief ("the tower height"): units ride the operators, converting by definition and reading in the larger unit, a length over a length is a ratio, and meters times meters refuses rather than invent a unit (0.000 -> 0.800 at 18.46x sd, zero wrong values, zero ordinary questions moved). That ceiling is now closed: an operand the store does not hold but can REACH goes through the chain machinery under the same guards as a comparative operand, marked where it is used (0.662 -> 1.000 at 3.64x sd, zero derived denials given a number). And "one matrix, one voice" became a measured line — every two-belief sum asked twice, through this rung and through the older combine path, required to agree on number and unit |
| **Answers that say what they are** | found by asking the shipped machinery a plain question: "the sum of the tower height and the keep height" answered "= 500" for two premises that both read in meters — the unit suffix had been attached to the conversion rather than to the result, so every same-unit sum, difference and larger/smaller verdict shipped a bare number (0.528 -> 1.000 at 1.87x sd, zero numeric values moved). The fix broke no pin, which is the tell: the case was pinned only for what the answer must NOT say |
| **Exact arithmetic** | "what is 3 + 4 * 5?" reads precedence and parentheses structurally and evaluates over exact rationals, not floats: 0.1 + 0.2 is 0.3, one third prints as 1/3 because 0.333… is a different number, and division by zero refuses. Measured against the standard option rather than an absence — the same reader in binary floating point — which printed a number that was not the answer 45 times and answered 8 undefined divisions, once reporting 126,100,789,566,373,888 for an expression whose divisor is zero (0.443 -> 1.000 at 3.69x sd) |
| **Modifier tolerance** | up to three library-unknown adjectives read away behind one unmoved residual floor, every dropped word named so the reading can be vetoed — three cap moves, the floor untouched, the distinction measured each time |
| **Revision accountability** | a change of mind is narrated (old belief named, retracted derivations counted), a near-key statement names the held base it sits beside, and agreements are never called revisions (0.000 -> 0.803 at 5.03x sd, zero false notices) |
| **The arithmetic capstone** | every math form above measured against one 5,818-fact colliding world: ten forms at 1.000, zero fabrications across all eight refusal families (unheld operand, denial, non-numeric, unit times unit, division by zero, percentage of nothing, unknown relation, zero to the zero), zero false bounds. The price line then found something: literal arithmetic cost 38.9 ms, of which the arithmetic itself is 0.016 ms — the Recall stage is ~90% of the price of a question that names no belief. Registered with its number, not yet acted on |
| **The density capstone** | every question form above measured against one ~10,000-fact colliding world: floors at 1.000, fabrication at zero across every decoy family, every price named (depth-4 chains 347 ms after the frontier clock, recall-backed forms ~1 ms) |

Where a mechanism failed its gate, that is in the book too: the shared-encoder
stage failed twice, honestly, and reordered the roadmap; hypervector retrieval
passed on structure and failed on similarity; a rehearsal mechanism and an
escalation heuristic were deleted for measuring nothing. The most recent
negative result is the most useful one: skipping memory lookup for questions
that are pure arithmetic made those questions 18x faster at perfect byte
parity — and made the whole session no faster at all (1497 ms -> 1530 ms),
because the paging it removed was warming the cache for every question that
followed. The price is the bucket cache, not the lookup; the mechanism is
switched off, kept, and explained beside its own switch.

## Quick start

**Windows:** double-click `UltraQuant.bat` — no packages, no environment.
**Linux/macOS:** `./ultraquant.sh` (desktop app with Tkinter, terminal UI
without).

```
python -m ultraquant.gui                   # desktop app: 10 tabs
python -m ultraquant.tui                   # the same surfaces over SSH
python -m ultraquant.interpreter.chat     # terminal chat
python -m ultraquant.forge.build --synthetic 64 --compare
python -m unittest discover -s tests      # 1809 tests, ~4 min
```

In the chat, try:

```
#####            <- paste a 5x5 glyph; it says what it sees, in its languages
:learn           <- the model surveys its own gaps and asks you questions
:learn research  <- it tries the web first (':online on'), quarantined
:learn panel ... <- or asks local models, counted by independent lineage
:study           <- one self-study cycle: survey, ask the panel, apply only
                    what independent voices corroborate
what is 3 + 4 * 5?   <- exact arithmetic; 0.1 + 0.2 is 0.3, and 1/3 stays 1/3
goal: the tower height and the bridge length    <- multi-step planning
:trace           <- the thought pipeline behind the last answer
:correct <cat> <text>   <- that query belonged elsewhere; say so
```

## Layout

```
ultraquant/
  quantum/     gates - state - circuit - ansatz - vqc - kernels - noise - search
  model/       ternary quantization - UltraQuantNet - the (failed, kept) encoder
  shards/      vault - sketch screen - router - budget - prototype consolidation
  experts/     per-category nets, paged on demand
  reason/      blackboard - discovery - planner - hypervectors - language
               inference (chains) - calculate (exact arithmetic)
  memory/      episodic/semantic stores - facts as bucket shards
  interpreter/ thought pipeline - chat CLI - stash quarantine - learning mode
  forge/       build libraries from scratch - deployment languages - seed facts
  native/      C++/CUDA accelerators - the learned dispatch scheduler
  storage/     NVMe-oF / Ceph / SAN backends - RAM tier - paged index
  experiments/ the gates: every capability's pre-registered measurement
tests/         1809 tests across 120 modules
```

The deep documentation is [ARCHITECTURE.md](ARCHITECTURE.md): design
principles, measured performance of every execution tier, the honest QPU
analysis (§7 — including why the data-loading wall is real), and the staged
path toward generality with every gate's verdict, pass or fail (§11).

## Settings persist

The GUI and TUI read a settings file on load and write it as things change, so
a RAM budget, storage location, device choice, forge defaults or chosen model
panel survive a restart. `:settings` in the TUI or chat CLI, and
**Session -> Where are my settings?** in the GUI, print the path and contents.

It is looked for in this order:

1. `$ULTRAQUANT_CONFIG` - an explicit path.
2. `./ultraquant.json` **if it already exists** - portable mode, so a copy on a
   USB stick keeps its settings with it. Never created implicitly.
3. `%APPDATA%\UltraQuant\settings.json` on Windows, or
   `$XDG_CONFIG_HOME/ultraquant/settings.json` elsewhere.

Credentials are deliberately **not** stored - the file is plaintext and may sit
in a checkout, so the BlueQubit token and array password stay in the
environment. A corrupt or unreadable file never stops the program starting: it
falls back to defaults and says what it could not use.

## Training the library from local models

```bash
python -m ultraquant.forge.train_from_llm uq_home --next
```

The same run is one click in the GUI: **Panel -> Distill next voice**,
with the cumulative ledger shown beside the button.

Each invocation trains from the **next untried voice, smallest model first** -
one model loaded at a time, never several. The budget is cumulative: each
category absorbs at most 12 distilled phrasings ever (the measured ceiling from
ARCHITECTURE.md §11.13), tracked in `vault/distilled.json`, and the run says
so when the catalogue is full. Held-out routing and a decoy check are measured every run - and a run that
makes the router claim unrelated queries is **rolled back in the same
transaction**, not merely reported.

## Rebuilding the native accelerators (optional)

Prebuilt Windows DLLs ship in `ultraquant/native/_bin/`. To rebuild:

```
powershell -ExecutionPolicy Bypass -File native\build.ps1 -Target all   # Windows
native/build.sh --target all                                            # Linux/macOS
```

Without them everything runs on the pure-Python tier — slower, identical
results.

## LM Studio and the LLMLS panel (optional)

If [LM Studio](https://lmstudio.ai) is running locally, the **Panel** tab in the
GUI — and `:panel` in the chat CLI, and the `panel` screen in the TUI — puts a
question to several open-source models you choose and reports what they agreed
on, weighted by **independent lineage** rather than headcount.

In the GUI the models are listed *under the voice they belong to*, and the
selection line updates as you pick: choose four Llama derivatives and it reads
`4 model(s) selected = 1 independent voice(s)` before you ask anything.

```bash
python -m ultraquant.interpreter.chat
```

```
:panel                                    # catalogue, grouped into voices
:panel qwen/qwen3-coder-30b openai/gpt-oss-20b ? what is the capital of Australia
```

It also answers the system's **own** questions. In the GUI, 'Ask the panel' on
the Learn tab uses whichever models are selected on the Panel tab; a corroborated
answer is filled into the answer box for you to review and submit, and an
uncorroborated one leaves the box empty and the question open. The CLI and TUI
print the exact command that would apply it. Nothing is ever applied
automatically - a panel that submitted its own answers would be an oracle.

The point is the accounting. A model and its own fine-tune agreeing is one
source, not two, and on a real catalogue nine chat models resolve to **four
independent voices**. Models are loaded on demand with a TTL so a large panel
does not need to be resident. Nothing a model says is believed: answers enter
the same quarantine as scraped web pages and must still earn promotion.

What this deliberately does **not** do is answer for the system. The
interpreter's replies come from the grammar it induced itself (ARCHITECTURE.md
§11.10); routing them through a chat model would make every measured claim in
§11 unfalsifiable while making the output look better. The related experiment —
whether a real embedding model routes better than lexical signatures — is
§11.11, and it **failed** its own control.

## The rules this codebase is built under

1. **The simulator is the reference** — accelerators are never semantics.
2. **Never load what you don't need** — store size is bounded by disk, not RAM.
3. **Recall reinforces** — the catalog reorganises around what is used.
4. **Found is not believed** — web content is quarantined until it earns belief.
5. **Gates before experiments** — thresholds are stated in advance, failures
   are published, and mechanisms that measure nothing get deleted.
