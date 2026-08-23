# UltraQuant — Development journal

Every unit of work since [ARCHITECTURE.md](ARCHITECTURE.md)'s staged
path was completed. Each entry is one capability, the falsifiable gate
written **before** it ran, and the verdict written **after** —
including the verdicts that were failures, which are kept rather than
deleted because a mechanism that did not work is a result about the
design, not an embarrassment about the attempt.

This file was split out of ARCHITECTURE.md at **99 entries**, when the
journal had grown to 4,198 lines against the architecture's 2,745 and
only seven of its sections appeared in any table of contents. Nothing
was edited in the move.

The architecture document keeps §11.1–§11.10: the assessment of what
this system is, the two things it was missing, and the six staged
gates. Those are claims about the design. Everything here is the
record of building against them.

**Sections appear below in the order they were written into the file,
which is not numeric — the index is sorted newest first, so use it.**

---

## Index

| § | unit |
|---|---|
| [11.109](#11109-stage-1-in-the-domain-it-was-told-to-try) | Stage 1, in the domain it was told to try |
| [11.108](#11108-the-edge-was-semantic-after-all) | The edge was semantic after all |
| [11.107](#11107-three-defects-measured-before-they-were-fixed) | Three defects, measured before they were fixed |
| [11.106](#11106-what-it-costs-to-change-your-mind) | What it costs to change your mind |
| [11.105](#11105-ultraquant-against-a-transformer-on-the-same-facts) | UltraQuant against a transformer, on the same facts |
| [11.104](#11104-reading-until-you-know-enough-failed-kept) | Reading until you know enough (failed, kept) |
| [11.103](#11103-an-encoder-that-refuses-its-own-output) | An encoder that refuses its own output |
| [11.102](#11102-a-prediction-layer-that-says-how-much-it-knows-failed-kept) | A prediction layer that says how much it knows (failed, kept) |
| [11.101](#11101-the-27-block-encoder) | The 27-block encoder |
| [11.100](#11100-the-operations-between-the-matmuls) | The operations between the matmuls |
| [11.99](#1199-a-glyph-for-every-imported-tensor-failed-kept) | A glyph for every imported tensor (failed, kept) |
| [11.98](#1198-weights-that-come-back-out-and-still-recognise) | Weights that come back out and still recognise |
| [11.97](#1197-the-yesno-family-natively) | The yes/no family, natively |
| [11.96](#1196-tensors-that-go-into-the-library) | Tensors that go into the library |
| [11.95](#1195-what-converting-a-transformer-costs-failed-kept) | What converting a transformer costs (failed, kept) |
| [11.94](#1194-one-copy-of-a-rule-that-has-to-agree) | One copy of a rule that has to agree |
| [11.93](#1193-a-claim-about-a-set-twice) | A claim about a set, twice |
| [11.92](#1192-two-spreads-that-converge-the-same-way) | Two spreads that converge the same way |
| [11.91](#1191-one-conversation-two-tiers-the-same-words) | One conversation, two tiers, the same words |
| [11.90](#1190-two-stores-that-remember-the-same-way) | Two stores that remember the same way |
| [11.89](#1189-the-same-sentence-from-a-different-language) | The same sentence, from a different language |
| [11.88](#1188-a-native-tier-that-has-to-agree) | A native tier that has to agree |
| [11.87](#1187-a-list-of-numbers-is-not-a-list-of-beliefs) | A list of numbers is not a list of beliefs |
| [11.86](#1186-the-question-that-needed-no-memory-failed-kept) | The question that needed no memory (failed, kept) |
| [11.85](#1185-the-arithmetic-capstone) | The arithmetic capstone |
| [11.84](#1184-rounding-is-a-request) | Rounding is a request |
| [11.83](#1183-asking-whether-the-arithmetic-holds) | Asking whether the arithmetic holds |
| [11.82](#1182-a-coin-that-always-said-no) | A coin that always said No |
| [11.81](#1181-a-root-that-has-no-value-still-has-a-place) | A root that has no value still has a place |
| [11.80](#1180-a-hundredth-part-taken) | A hundredth part, taken |
| [11.79](#1179-an-operand-the-store-can-reach) | An operand the store can reach |
| [11.78](#1178-arithmetic-over-what-is-believed) | Arithmetic over what is believed |
| [11.77](#1177-the-unit-that-fell-off) | The unit that fell off |
| [11.76](#1176-arithmetic-that-is-exactly-right) | Arithmetic that is exactly right |
| [11.75](#1175-the-comparative-word-carries-its-attribute) | The comparative word carries its attribute |
| [11.74](#1174-two-statements-joined-are-two-statements) | Two statements joined are two statements |
| [11.73](#1173-one-grammar-for-teaching-and-asking) | One grammar for teaching and asking |
| [11.72](#1172-the-relation-stated-once-distributes) | The relation stated once distributes |
| [11.71](#1171-conjunctive-statements-teach-all-their-facts) | Conjunctive statements teach all their facts |
| [11.70](#1170-declining-a-derivation-names-its-premises-the-family-whole) | Declining a derivation names its premises: the family whole |
| [11.69](#1169-no-is-testimony-too-doubt-spoken-never-absence) | No is testimony too: doubt spoken, never absence |
| [11.68](#1168-yes-reaches-the-stored-fact-testimony-at-the-surface) | Yes reaches the stored fact: testimony at the surface |
| [11.67](#1167-compound-disjuncts-compare-whole) | Compound disjuncts compare whole |
| [11.66](#1166-choices-the-held-disjunct-answers-by-name) | Choices: the held disjunct answers by name |
| [11.65](#1165-what-happened-the-retraction-askable--the-tense-family-closes) | What happened: the retraction, askable — the tense family closes |
| [11.64](#1164-what-was-belief-history-askable--and-never-invented) | What was: belief history, askable — and never invented |
| [11.63](#1163-the-neighbor-named-adjacency-spoken-never-merged) | The neighbor named: adjacency spoken, never merged |
| [11.62](#1162-the-third-adjective-the-cap-is-policy-the-floor-is-safety) | The third adjective: the cap is policy, the floor is safety |
| [11.61](#1161-the-fifth-fact-the-second-limit-moved-by-measurement) | The fifth fact: the second limit moved by measurement |
| [11.60](#1160-the-drowned-bridge-that-wasnt-a-negative-result-recorded) | The drowned bridge that wasn't: a negative result, recorded |
| [11.59](#1159-the-depth-clock-licensed-by-measurement-held-to-parity) | The depth clock: licensed by measurement, held to parity |
| [11.58](#1158-the-density-revisit-the-sessions-arc-at-scale) | The density revisit: the session's arc at scale |
| [11.57](#1157-aggregates-the-family-counted-totalled-and-averaged) | Aggregates: the family counted, totalled, and averaged |
| [11.56](#1156-superlatives-a-claim-about-a-set-with-the-set-named) | Superlatives: a claim about a set, with the set named |
| [11.55](#1155-derived-operands-compare-three-families-in-one-answer) | Derived operands compare: three families in one answer |
| [11.54](#1154-comparatives-a-verdict-that-was-computed-refusals-that-are-named) | Comparatives: a verdict that was computed, refusals that are named |
| [11.53](#1153-a-change-of-mind-said-out-loud) | A change of mind, said out loud |
| [11.52](#1152-the-fourth-fact-a-stated-limit-moved-by-measurement) | The fourth fact: a stated limit moved by measurement |
| [11.51](#1151-why-the-trail-is-the-answer-askable--and-never-invented) | Why: the trail is the answer, askable — and never invented |
| [11.50](#1150-polar-questions-answered-through-chains-the-family-closes) | Polar questions answered through chains: the family closes |
| [11.49](#1149-denials-speak-at-terminals-and-only-at-terminals) | Denials speak at terminals, and only at terminals |
| [11.48](#1148-polarity-belief-of-absence-inhibitory-everywhere-honest-aloud) | Polarity: belief-of-absence, inhibitory everywhere, honest aloud |
| [11.47](#1147-depth-three-fact-chains-and-the-three-layers-the-gate-indicted) | Depth: three-fact chains, and the three layers the gate indicted |
| [11.46](#1146-two-adjectives-may-decorate-the-cap-moves-the-floor-stays) | Two adjectives may decorate: the cap moves, the floor stays |
| [11.45](#1145-the-vram-tier-the-storage-split-completes-at-parity-first) | The VRAM tier: the storage split completes, at parity first |
| [11.44](#1144-plasticity-attestation-earns-rank-and-nothing-more) | Plasticity: attestation earns rank, and nothing more |
| [11.43](#1143-self-study-the-loop-that-closes-its-own-gaps-shipped-on-a-zero) | Self-study: the loop that closes its own gaps, shipped on a zero |
| [11.42](#1142-unit-conversion-the-refusal-narrowed-nothing-widened) | Unit conversion: the refusal narrowed, nothing widened |
| [11.41](#1141-containment-corroboration-the-bounded-exception-and-the-first-learned-facts) | Containment corroboration: the bounded exception, and the first learned facts |
| [11.40](#1140-the-first-library-training-run-three-panel-defects-and-an-honest-zero) | The first library-training run, three panel defects, and an honest zero |
| [11.39](#1139-the-suggester-goes-live-the-boundary-translates-the-core-still-refuses) | The suggester goes live: the boundary translates, the core still refuses |
| [11.38](#1138-the-density-gate-convergence-survives-three-orders-of-magnitude) | The density gate: convergence survives three orders of magnitude |
| [11.37](#1137-the-paraphrase-gate-the-language-wall-measured-the-hybrid-tradeoff-quantified) | The paraphrase gate: the language wall measured, the hybrid tradeoff quantified |
| [11.36](#1136-adjective-tolerance-one-word-may-decorate-never-substitute) | Adjective tolerance: one word may decorate, never substitute |
| [11.35](#1135-compound-questions-sequential-attention-honest-halves-hygienic-asks) | Compound questions: sequential attention, honest halves, hygienic asks |
| [11.34](#1134-the-ladder-the-systems-own-hints-steer-its-teaching-at-exactly-minimal-cost) | The ladder: the system's own hints steer its teaching, at exactly minimal cost |
| [11.33](#1133-curiosity-a-refused-inference-asks-for-exactly-what-it-was-missing) | Curiosity: a refused inference asks for exactly what it was missing |
| [11.32](#1132-the-validator-cannot-be-tightened-into-the-reader-and-the-shelf-gains-a-resident) | The validator cannot be tightened into the reader, and the shelf gains a resident |
| [11.31](#1131-consolidation-a-confirmed-thought-becomes-memory-and-revision-takes-it-back) | Consolidation: a confirmed thought becomes memory, and revision takes it back |
| [11.30](#1130-inference-by-spreading-activation-the-library-thinks-and-the-gate-holds) | Inference by spreading activation: the library thinks, and the gate holds |
| [11.29](#1129-a-battery-against-the-live-chat-surface-and-six-defects-fixed) | A battery against the live chat surface, and six defects fixed |
| [11.28](#1128-the-blind-reader-confirmed-the-last-suspect-and-the-chain-ends-with-a-mechanism) | The blind reader confirmed the last suspect, and the chain ends with a mechanism |
| [11.27](#1127-load-dies-too-and-the-variant-chain-closes-on-the-data-itself) | Load dies too, and the variant chain closes on the data itself |
| [11.26](#1126-one-distribution-after-all-1125s-diagnosis-dies-and-the-suspect-is-load) | One distribution after all: §11.25's diagnosis dies, and the suspect is load |
| [11.25](#1125-the-data-lever-measured-negative-and-the-diagnosis-is-distribution) | The data lever measured negative, and the diagnosis is distribution |
| [11.24](#1124-part-based-features-measured-nothing-and-the-candidate-list-is-spent) | Part-based features measured nothing, and the candidate list is spent |
| [11.23](#1123-the-adoption-gate-1122s-pass-stops-at-the-shape-it-was-measured-on) | The adoption gate: §11.22's pass stops at the shape it was measured on |
| [11.22](#1122-capacity-was-the-binding-constraint-and-scale-normalisation-measured-useless) | Capacity was the binding constraint, and scale normalisation measured useless |
| [11.21](#1121-the-translation-capable-representation-was-built-and-the-problem-moved) | The translation-capable representation was built, and the problem moved |
| [11.20](#1120-drawing-distillation-reached-the-experts-and-failed-on-their-features) | Drawing distillation reached the experts, and failed on their features |
| [11.19](#1119-the-storage-split-made-real-and-training-one-voice-at-a-time) | The storage split made real, and training one voice at a time |
| [11.18](#1118-correcting-a-route-and-a-claim-that-had-to-be-withdrawn) | Correcting a route, and a claim that had to be withdrawn |
| [11.17](#1117-reinforcement-gated-on-confirmation) | Reinforcement gated on confirmation |
| [11.16](#1116-self-reinforcement-entrenches-errors-and-a-second-training-run-proved-it) | Self-reinforcement entrenches errors, and a second training run proved it |
| [11.15](#1115-the-router-could-not-say-not-mine) | The router could not say "not mine" |
| [11.14](#1114-a-context-window-in-memory-with-disk-as-the-reference) | A context window in memory, with disk as the reference |
| [11.13](#1113-distilling-a-transformer-into-the-router-offline) | Distilling a transformer into the router, offline |
| [11.12](#1112-llmls-a-panel-of-local-models-and-what-their-agreement-is-worth) | LLMLS: a panel of local models, and what their agreement is worth |
| [11.11](#1111-an-external-encoder-was-tried-and-the-gate-failed-on-its-own-control) | An external encoder was tried, and the gate failed on its own control |
---

### 11.11 An external encoder was tried, and the gate failed on its own control

[§11.5](ARCHITECTURE.md#115-stage-1-was-run-and-it-failed) failed and named the reason: for
stroke patterns **raw pixels were already close to an ideal representation**, so
compressing them could only lose. It also named the condition under which a
learned representation *should* pay — "a perceptual domain where the raw features
are a poor representation". Text routing by lexical overlap is exactly that
domain: "a box outline" and "a square" share no token, so a signature built from
tokens cannot connect them in principle, not merely in practice.

LM Studio makes a real embedding model available locally over HTTP
(`ultraquant/interpreter/lmstudio.py`), so the mechanism §11.5 said was needed
could be tested rather than argued about. `ultraquant/experiments/embed_gate.py`
holds the gate, written before the run: on paraphrased queries the embedding
route must beat the lexical route by **more than the seed-to-seed standard
deviation** — the §11.5 criterion verbatim — and it must not win the control.

**The first control was at ceiling, and that is the finding.** The control used
*verbatim* queries, where the query is the stored text. Lexical scores 1.000
there, embeddings tied at 1.000, and the control was recorded as held. It was
not held: **at ceiling a control cannot fail in the direction it exists to
detect**, because nothing scores above 1.0. That is precisely the defect that
invalidated §11.5's first run, arriving from the opposite side. Under it, this
reported a clean PASS at 5.66× seed sd.

The control was rebuilt as a **partial-overlap** condition — queries sharing some
vocabulary with their category, so lexical lands mid-range with room above it and
a merely-better classifier has somewhere to show that.

| condition | lexical | embedding | delta |
|---|---:|---:|---:|
| paraphrased (disjoint vocabulary) | 0.300 | 0.833 | **+0.533** |
| partial overlap (**control**) | 0.912 | 1.000 | **+0.087** |
| verbatim (sanity, at ceiling) | 1.000 | 1.000 | +0.000 |

*8 seeds; paraphrase seed sd 0.094, margin ratio 5.66×; control sd 0.033.*

**FAIL.** The paraphrase margin is large and clears the §11.5 criterion
comfortably. But embeddings also win the control by +0.087, which is 2.6× the
control's own noise — so part of the advantage is capacity, not meaning, and
under the criterion written before the run the paraphrase margin cannot be
attributed to semantics.

**The rescue that was declined.** A pure-capacity account predicts a roughly
*equal* edge in both conditions, and the paraphrase edge is 6× the control edge.
A difference-in-differences would read **+0.446** and pass easily — and is
probably the better-specified test, since subtracting the control is what a
control is for. But it was formulated after seeing the numbers, and adopting a
criterion because it converts a fail into a pass is the same goalpost move §11.5
refused when a +0.014 effect's interval cleared zero. So the gate failed. The
difference-in-differences criterion is now written down for a future run **on
categories not used here**, where it will be pre-registered rather than
retrofitted, and only that run can pass it.

The cost is also on the record: 2.9 ms per embedded string against 0.015 ms for a
token-set intersection, a **197× ratio** that any future pass would still have to
justify.

### 11.12 LLMLS: a panel of local models, and what their agreement is worth

`ultraquant/interpreter/llmls.py` lets the user choose which open-source models
sit on a panel; LM Studio loads them on demand, each is asked the same question
in isolation, and what they say enters the contemporary stash — quarantined,
like any scraped page. The models are **sources being interrogated**, never the
system's voice: routing answers through a chat model would make every claim in
§11 unfalsifiable while making the demo look better, which is the worst available
trade.

**The mechanism that makes a panel worth more than a model is independence, and
independence cannot be proved from metadata.** The catalogue on this machine
shows why neither available field works alone:

| model | `arch` | `publisher` |
|---|---|---|
| `qwen/qwen3-coder-30b` | qwen3moe | qwen |
| `qwen/qwen3-coder-next` | qwen3**next** | qwen |
| `openai/gpt-oss-20b` | gpt-oss | openai |
| `openai-gpt-oss-20b-abliterated-…` | gpt-oss | **DavidAU** |
| `deepseek-coder-33b-instruct` | llama | TheBloke |
| `codellama-34b` | llama | TheBloke |

The two Qwens report *different* architectures and are obviously one lineage —
publisher catches them. The gpt-oss pair has different publishers and is one
lineage, a fine-tune of the other — architecture catches that. Meanwhile
`arch=llama` and `publisher=TheBloke` both lump DeepSeek-Coder with CodeLlama,
which are separate trainings sharing an architecture and a quantizer.

So `lineage_key` deliberately **over-groups**: any shared architecture,
publisher, or name stem collapses models into one voice. It will sometimes cost
the panel corroboration it had genuinely earned. That direction is chosen because
the opposite error *manufactures evidence*, and every count is reported as a
**lower bound on correlation** rather than a proof of independence.

On the real catalogue this resolves **9 chat models to 4 independent voices** —
the number that actually governs how much a unanimous panel is worth, and one
that is invisible without the accounting.

**A measurement error caught by running it.** The first live run over three
independent voices asked for the time complexity of binary search. All three
answered O(log n) — unanimous — but two wrapped it in different prose, positions
were compared by exact string match, and the panel reported **2 of 3 voices**. On
"best language for beginners" all three said Python and it reported **1 of 3**.
An agreement measure that misses unanimity is not conservative, it is broken.

The fix was not a semantic-similarity heuristic. Deciding that two
differently-worded sentences mean the same thing is the hard problem, and a
heuristic guessing at it would manufacture consensus — the one error this system
must not make. Instead the panel **constrains the reply format** (`TERSE`: the
bare fact, at most eight words), which makes exact matching a meaningful
comparison, and warns explicitly when replies come back as prose that the
reported agreement is a floor.

A second live run over the same three voices, same questions:

| question | before | after |
|---|---:|---:|
| time complexity of binary search | 2 of 3 | **3 of 3** (`o log n`) |
| capital of Australia | — | **3 of 3** (`canberra`, not Sydney) |
| best language for beginners | 1 of 3 | 2 of 3 |
| how many moons does Mars have | — | 1 of 3 → **3 of 3** |

The moons row needed one further change. Three models answered `two moons`,
`two` and `2` — unanimous, reported as a three-way split. That is not the hard
semantic problem, it is spelling, so normalisation was extended to **orthography
only**: number words, filler, and words the question itself supplied (every
model echoes the prompt, and an echo is not a position). The line is drawn
explicitly and tested — rewriting `two` to `2` is the same class of operation as
lowercasing, while deciding `fast` and `efficient` are one position would be a
claim about meaning and is refused. `tests/test_lmstudio.py` asserts both halves,
so the refused capability cannot creep in later.

**The control that matters ran live.** Two Qwen models — one lineage — were asked
the capital of Australia. Both answered `canberra`, verbatim agreement, and the
panel reported **not corroborated, 1 of 1 voices**. That is the central claim
demonstrated rather than argued: two models agreeing is not two sources, and no
amount of asking converts one voice into evidence.

**Every surface reaches it.** `:panel` in the chat CLI, a `panel` screen in the
TUI, and a **Panel tab** in the GUI. The GUI is where the accounting is most
visible: models are listed *under the voice they belong to*, and the selection
line updates as you pick — choose four Llama derivatives and it reads
`4 model(s) selected = 1 independent voice(s)` before a question is asked, rather
than returning a mysteriously uncorroborated result afterwards. Everything slow
runs on the worker thread, since a panel question costs 8–25 s.

**An interaction bug, found by running the pieces together.** `TERSE` and the
stash pull in opposite directions, and neither measurement alone showed it. The
GUI panel agreed 3-of-3 that gold is `au`, reported it corroborated — and
quarantined **nothing**. `add_page` requires 20–420 characters, on the sound
grounds that a two-character fragment of a web page is not a claim; a terse
answer is exactly that fragment. Constraining replies to make them comparable
had silently disabled the quarantine path, with the checkbox still ticked.

The fix stores the question alongside the answer (`_as_claim`), both verbatim.
What was deliberately *not* done is asking a model to phrase the agreed fact as
a sentence — that would put wording nothing corroborated into an entry carrying
a corroborated claim's provenance.

**Wired into self-learning, offered rather than applied.** The panel answers the
system's *own* questions — the same shape as researching the web, and the same
rule. In the GUI, 'Ask the panel' on the Learn tab uses the models selected on
the Panel tab; a corroborated answer is written into the answer box for the user
to read, edit and submit, and an uncorroborated one is reported with the box left
**empty**. The chat CLI prints the exact `:learn answer <text>` that would apply
it; the TUI does the same. Nothing is ever applied automatically: a panel that
submitted its own answers would be an oracle, which is the arrangement this whole
module refuses, and that difference has to be visible in the UI rather than only
in a docstring.

Both branches were run against three independent voices. `What is the chemical
symbol for gold?` → 3 of 3 on `au`, box filled, stash entry `staged | What is
the chemical symbol for gold: au`. The system's own untrained-category question
`Tell me something about arithmetic` → two different answers and one model with
nothing usable, **not corroborated**, box left empty, question still open. The
second is the more informative result: an open-ended prompt has many right
answers, so consensus is the wrong instrument for it, and the panel reporting no
agreement is the honest reading rather than a failure.

**Three ways a question can defeat a panel, found by using it.** A user ran
`what is code?` and got a split; unpicking it turned up three different problems
wearing the same result.

*Under-specification, which was fixed.* gpt-oss answered "set of instructions for
a computer" and Qwen "secret communication system" — **both correct**, about
different senses of a polysemous word. The system already held what settles it:
it asked because it had seen `code` six times, and those sentences say which
sense it meant. `_unknown_terms` was counting the occurrences and discarding the
utterances. It now keeps up to four, `grounded_prompt` attaches them, and both
models moved onto the programming sense.

*Token starvation, which was a bug in the request.* Grounded, gpt-oss returned an
**empty string**, which the panel recorded as "no usable answer" — i.e. as the
model having nothing to say. The `usage` field said otherwise: at
`max_tokens=60`, **51 of 60 completion tokens went to reasoning**, leaving nine.
At 200 the identical request answered. A reasoning model's thinking is billed
against the same budget, so `max_tokens` was never the length control — `TERSE`
is — and the default is now 400. Truncation and abstention are reported
separately, because they need different fixes.

*Definitional answers, which remain unsolved.* With both fixed, the two voices
returned "sequence instructions executed computer" and "computer instructions
written programming language". Same claim, different words, still scored a split.
Positions are compared exactly, and deciding two phrasings mean the same thing is
the semantic-equivalence problem this module refuses to guess at.

So the panel corroborates cleanly where an answer has a canonical form — `au`,
`canberra`, `2`, `frank herbert`, `o log n` — and poorly on definitions, which
legitimately admit many wordings. That is a limit, not a tuning problem.
`Consensus.overlap_note` states it in the output so "not corroborated" is not
misread as "the models disagree", and it is **reported and never credited** —
[§the entropy black box](ARCHITECTURE.md#the-entropy-black-box)'s monobit rule, applied here for
the same reason. Its threshold is calibrated on six observed splits that separate
at 0.29/0.20 against 0.10/0.00/0.00/0.00; six is a small calibration and it will
misfire, which is tolerable *only* because the note moves no verdict.

**A resource bug found in `lms ps`.** `lms load` on an already-resident model
does not no-op — it loads a **second instance** under a `:2` identifier. Two
panel runs had left 61.5 GB resident across `gpt-oss-20b`, `gpt-oss-20b:2`,
`qwen3-coder-30b` and `qwen3-coder-30b:2`, half of it duplicates this code
created. `load()` now checks residency first.

This is not a capability gate and is not claimed as one. It is an integration
with an accounting attached, and the accounting is the part worth having.

### 11.13 Distilling a transformer into the router, offline

[§11.11](#1111-an-external-encoder-was-tried-and-the-gate-failed-on-its-own-control)
left a real trade open. Embeddings routed paraphrases at **0.833** against
lexical **0.300**, but cost **2.9 ms per string against 0.015 ms** — a 197x tax
on every query forever, plus a model resident to serve it. On a machine where
VRAM is the binding constraint that is the wrong way round.

`ultraquant/forge/distill.py` pays it **once, offline, and never again**: ask a
local model for query phrasings, teach them into the router's existing
token-weight table, and query time stays in microseconds with nothing new
resident. The transformer becomes a *teacher consulted at build time*, not a
dependency at inference time. It transfers **labelled examples, not weights** —
the only currency a transformer and a token-weight table share.

**Getting any output at all was the first problem, and it is worth recording.**
Asked for six short phrasings, `qwen3.6-35b-a3b` spent **400 of 400** completion
tokens reasoning and emitted nothing; at **1200 of 1200** it did the same. This
is *not* the starvation §11.12 fixed by raising the budget — raising it only
buys more thinking. `reasoning_effort="none"` returned the answer in 42 tokens
with zero reasoning. A `/no_think` suffix, `reasoning_effort="low"` and
`chat_template_kwargs={"enable_thinking": False}` were each tried and had **no
effect at all**. Generation went from 99 s and 0 phrasings to **6 s and 12**.
The two failures now report differently, because they need different fixes.

**Two mechanisms were run against the pre-registered gate. The first failed.**

| mechanism | held-out paraphrases | margin | decoy control |
|---|---:|---:|---:|
| raw phrasings | 0.035 → 0.396 | 6.31x sd | **1.000 → 0.133** |
| content tokens only | 0.028 → 0.090 | 1.79x sd | 1.000 → 1.000 |

The first looks like a triumph and is a fraud. Natural phrasings are dense in
function words, so `what`, `how`, `do` and `the` acquired weight for every
taught category, and *everything* started matching: "how do I fix a leaking tap"
was pulled into the taught set, and 87% of decoys were misrouted. **The router
had learned to say yes, not to know more.** A ceilinged control would have
reported a clean pass at 6.31x seed variance — which is precisely the defect
§11.11 was caught making, arriving here in a form that would have been believed.
This control starts at 1.000 and can only fall, so it had room to condemn the
result, and it did.

Filtering to tokens that actually name something passes: **+0.062 at 1.79x seed
sd with the control held exactly at 1.000**.

**What passing does not mean.** The improvement is real, controlled and
reproducible, and it is *small*: 2.8% to 9.0%. Both numbers are poor, and
§11.11 measured the embedding route at 0.833 on the same kind of task. So
distillation recovers a **small fraction** of what a resident embedding model
delivers. It is the right trade when VRAM is the binding constraint and query
latency must stay in microseconds; it is not a substitute for embeddings where
accuracy is what matters. Reading "PASS" as "routing now works" would be a
misreading.

**It does not scale, and that is the finding that matters most.** The obvious
next move after a small pass is to generate more data. Measured, that breaks it:

| volume | before | after | margin | decoy control |
|---|---:|---:|---:|---:|
| 48 phrasings (12 per category) | 0.028 | 0.090 | 1.79x | **1.000** |
| 117 phrasings (30 per category) | 0.000 | 0.492 | 8.93x | **0.767** |

At the larger volume the margin looks *spectacular* — half the held-out
paraphrases routed, nearly 9x seed variance — and a quarter of unrelated decoys
are back to being misrouted. Content filtering **delayed** the yes-machine
effect; it did not remove it. Enough learned tokens, however carefully chosen,
still make every taught category match everything.

`SAFE_PHRASINGS_PER_CATEGORY = 12` therefore caps generation, as a **measured
limit rather than a tuning knob**. Raising it makes the reported margin go up
and the result stop meaning anything — which is precisely the trap the control
exists to catch, now demonstrated twice in the same section. Nothing here
establishes that the true ceiling is 12 rather than 15 or 20; only that 12 holds
and 30 does not.

**The margin was also inflated by the wrong standard deviation.** The gate used
`statistics.pstdev` over its splits — population sd, which divides by n rather
than n-1 and so understates the spread of a *sample*. Corrected to `stdev`, the
reported margin falls from 1.96x to **1.79x**. It still clears the threshold,
and it was never as far above it as published.

Two more gate defects came out of the same review. Zero seed variance reported
`margin_ratio = inf`, so a run whose splits all gave the identical delta — i.e.
one that varied nothing — was an **automatic pass**; it now fails with that
stated as the reason. And the printed control legend read "must not rise" while
the code fails the gate on a *fall*, which is exactly backwards for a control
measuring decoys correctly left alone.

**A caveat the gate does not yet cover.** Generation is not deterministic even
at temperature 0: three identical runs at 48 phrasings gave 0.090, 0.111 and
0.118. The margin therefore carries run-to-run variance *on top of* the seed
variance the gate measures, and the gate does not account for it. At 1.79x that
is not a comfortable distance from the 1.0 threshold.

**Run against the deployed library.** `ultraquant/forge/train_from_llm.py`
points this at a real session rather than the gate's synthetic categories, and
two judgements it has to make are worth recording.

A forged library carries 24 categories named `family_000`…`family_023` whose
keywords are `sym`, `0`, `1`. Asking a model "different ways to ask about
family_007" produces confident nonsense, which would then be taught to the
router as though it meant something — so they are skipped, by name **and** by
checking the vocabulary is descriptive, and reported as skipped rather than
quietly dropped. On the deployed home: **8 real categories trained, 46
phrasings, 24 synthetic families skipped**, held-out routing 0.109 → 0.174.

The `--dry-run` flag was wrong on its first outing and the bug is instructive:
it skipped `router.save()`, but `CategoryRouter.learn()` also calls
`vault.reinforce()`, which writes the shard catalog immediately. The "dry" run
had genuinely trained the library, and the next run started from the moved
baseline — visible only because the second run's *before* was 0.174 rather than
0.109. A dry run now works on a copy of the home.

**One thing the run surfaced that distillation did not cause.** On the deployed
library only **0.200** of unrelated decoy queries are correctly left alone —
before any training. With 32 categories registered, the router already claims
four unrelated queries in five. Distillation held that number exactly (0.200 →
0.200), so it neither caused nor worsened it, but the router's baseline
willingness to answer is a larger problem than anything measured in §11.13 and
is not yet behind a gate.

### 11.15 The router could not say "not mine"

§11.13's training run surfaced it without causing it: on the deployed library,
of five queries belonging to **none** of its categories, the router declined
**one**. "what time does the train leave" routed to `crosses`. So did "the
weather forecast for tomorrow", and "how do I fix a leaking tap". None matched a
single base keyword.

`route()` admits any category scoring above zero, so one weighted token is
enough to claim a query. The question was where the weight came from, and the
answer turned out to be **three separate stores with the same defect**, each of
which had to be fixed on its own — and each of which made the previous fix look
insufficient rather than wrong.

**One — learned weights.** `CategoryRouter.learn` took every distinct token of
whatever text it was given, so `the`, `a`, `is` and `what` accumulated weight for
whichever category the input happened to route to. **21% of all learned weight
on the deployed router sat on stopwords**, led by `is` at 1.05.

This is the same defect §11.13 fixed in `distil_into_router`, where the reasoning
was explicitly that bulk-generated phrasings were the volume case and *"a single
user typing one query at a time contributes far less generic weight, which is
why `CategoryRouter.learn` is left alone"*. That reasoning was wrong, and the
measurement is what says so: **ten ordinary learning turns are enough**.

**Two — vault associations.** With the router's own table cleaned, decoys still
routed, scoring 0.7 to 1.2 on **vault association alone** with no base hit and no
learned weight. `learn()` reinforces the vault with the same token list, so the
durable store had accumulated the same stopwords.

**Three — registered keywords.** With both cleaned, two decoys survived: `world`
shipped with `what`, `who` and `where` as *base* keywords. Registering a question
word is how a category comes to claim every question.

One rule — `_informative` — now applies at every point a token can enter, and
existing libraries are repaired on load rather than only new ones being spared.

**The gate, and the control that makes it mean anything.** Abstention is trivial
to maximise: a router that returns nothing scores 1.000 on decoys. So genuine
routing is measured alongside it and a fall fails the gate outright. A test
asserts the control *can* fail, by building a router whose whole vocabulary is
stopwords — perfect abstention, zero routing.

| | before | after |
|---|---:|---:|
| decoys declined (gate, 8 seeds) | 0.025 | **0.975** (+0.950, **12.57x** seed sd) |
| genuine routed (**control**) | 1.000 | 1.000 |
| decoys declined (**deployed library**) | 0.200 | **1.000** |
| genuine routed (deployed library) | — | 0.875 |
| learned weight discarded on repair | — | 36.5 |

The deployed library's one miss is **not** a regression: "add these two numbers"
goes to `crosses` because `arithmetic` registers `number` and the query says
`numbers`, while `crosses` had already accumulated `two`. Both pre-date this
change. A singular/plural vocabulary gap is a separate problem and is not fixed
here.

**The singular/plural gap, fixed separately.** §11.15 left one genuine query
misrouted: `arithmetic` registers `number`, the query said `numbers`, and the
category scored **zero** on a query obviously its own. `normalize_token` folds
regular plurals at registration, learning and routing, so whichever spelling
goes in, both match.

It is a *plural folder*, not a stemmer, and that restraint is the design.
Porter-style stemming also strips `-ing`, `-ed` and `-ation`, merging words that
mean different things — which, in a router that had just been taught not to
over-claim, would widen matching in precisely the direction that had regressed.
Words that merely end in `s` are protected: `class`, `status`, `analysis`,
`chaos` and `bus` survive untouched, where a stemmer turns `class` into `clas`
and it stops matching itself.

Measured: **0.500 without folding, 1.000 with it** — not 0.000 without, because
several categories happen to register both spellings (`frame` *and* `frames`),
which masks the gap wherever someone thought to do it. The failures are where
nobody did. Abstention was unchanged at 0.975, which is the control that matters
for a widening change.

**And it did not fix the deployed library's miss, for a different reason worth
recording.** "add these two numbers" still goes to `crosses`. Folding worked —
`arithmetic` now scores 1.10 where it scored 0 on base overlap — but `crosses`
scores **1.75**, on learned `two` at 0.30, learned **`number` at 0.15**, and 1.3
of vault association.

`crosses` learned `number`. It won that query wrongly once, and `learn()`
reinforces whatever category the pipeline routed to, so the error taught itself
and became more likely to win next time. That is a **feedback loop in the
learning rule**, not a vocabulary gap, and it is not fixed here — a router that
learns from its own routing has no way to notice it was wrong.

### 11.16 Self-reinforcement entrenches errors, and a second training run proved it

§11.15's plural fix worked and did not rescue the deployed library, which was the
clue. `ultraquant/experiments/reinforcement_drift.py` measures why.

`thoughts.py:801` reinforces whichever category the pipeline routed to, at
`delta=0.05`, with nothing checking the route was right. The obvious guess is
that accuracy decays. **It does not** — measured over 60 rounds, accuracy is
0.400 before and 0.400 after, with or without reinforcement. Reinforcement
cannot make a wrong answer wronger.

**What it does is make errors resistant to correction.** The margin by which the
wrong category beats the right one grows **0.60 → 2.33** when the router
reinforces itself, and stays flat at 0.60 when it does not: **3.9x harder to
overturn**, from nothing but routing normally.

That explains §11.15's leftover exactly. Plural folding gave `arithmetic` a full
point of base overlap — but `crosses` had entrenched to 1.75 while `arithmetic`
reached 1.10. The fix was real and arrived after the error had been reinforced
past it.

**A second training run then demonstrated it live.** Retraining the deployed
library from the same teacher:

| | first run | second run |
|---|---:|---:|
| categories trained | 8 (46 phrasings) | 8 (46 phrasings) |
| decoys left alone | 0.200 → 1.000 | 1.000 → **1.000** |
| genuine routed | 0.875 | **0.750** |
| held-out routing | +0.065 | **−0.022** |

Both new misses go to `crosses` — the same sink. Continued distillation into a
router that reinforces its own choices **degraded** it, while abstention held
perfectly, which is the §11.15 fix proving robust in the same run that showed
this one is not.

So the honest position on repeat training: it is not idempotent and it is not
safe to run repeatedly against the same library. §11.13 already found the
mechanism does not scale with *volume*; this is the same limit showing up in
*repetition*.

**Why the loop is not fixed here.** Correcting it needs a signal for whether a
route was right, and the pipeline has none — the router's own choice is the only
thing available at that point, which is the circularity itself. The supervised
calls are untouched and legitimate: `learning.py` and `selflearn.py` reinforce
from a *user's* answer, which is independent evidence. Replacing the
unsupervised call is a design question deserving its own gate, not a patch
applied while its consequences are still being measured.

**A reporting defect found on the way.** The second training run first came back
with "only 0 usable phrasing(s)" for every category, which points at the prompt.
The teacher was fine: it needed **65 s per category** against the client's 120 s
default, and `generate_phrasings` catches a per-teacher failure internally, so a
timeout surfaced as an empty result. The timeout is now 600 s and the real
reason is reported instead of the symptom.

### 11.17 Reinforcement gated on confirmation

§11.16 measured the loop and left it open because fixing it needs a signal the
pipeline appeared not to have. It has a weaker one that is good enough: not
whether a route was *correct*, but whether the machinery the route reached did
anything.

`_route_confirmation` accepts exactly two signals, and both come from the routed
experts: an expert **placed the input** (`prediction` set, not flagged
unfamiliar), or the blackboard **composed a reading**. Absent either, the turn
produced nothing through that route and it is not reinforced — the trace says
`not reinforced (geometry): unconfirmed route` rather than staying silent.

**Three signals had to be removed, and running it is what caught them.** The
first version also accepted a recalled fact, a code result and an advanced plan.
None depends on the route: `Recall` runs **before** `Route` in the pipeline, so a
fact is found wherever the text routes, and code and planning are driven by
intent rather than category. Live, "what shape is this glyph" reinforced
`crosses` purely because a fact happened to be recalled — the original defect
wearing a confirmation signal that confirmed the wrong thing. A test asserts the
pipeline order, so the reason cannot rot.

| | unconditional | confirmation-gated |
|---|---:|---:|
| wrong-answer margin after 40 rounds | 2.33 | **0.60** (+1.73, **7.75x** seed sd) |
| paraphrase transfer (**control**) | 1.000 | 1.000 |
| total learned weight | 8.38 | 3.19 |

**The control had to be rebuilt before it meant anything — the third time.** Its
paraphrases were "crossing marks on the page" and "a boxed square shape", which
contain `mark` and `square`: both *registered* keywords. Verified against a
router with no learning at all, both routed correctly, so the control read 1.000
for a mechanism that had learned nothing. The rebuilt paraphrases share only a
token registered nowhere (`hatching`, `bordered`) and score **0.000** without
learning, which a test now asserts.

**It is prophylactic, not curative — and the obvious remedy is wrong.** Weight
already accumulated stays accumulated, so the deployed library keeps its two
entrenched `crosses` errors at 0.750 genuine routing. The tempting fix is a
clean redeploy. Measured, a fresh deploy scores **0.500** — *worse*. The
accumulated learning is net positive: entrenchment cost two queries while
learning gained four over bare vocabulary. So the remedy for those two is
targeted supervised correction through `:learn`, which was always the legitimate
path, not discarding the library's experience.

### 11.18 Correcting a route, and a claim that had to be withdrawn

§11.17 said the remedy for the deployed library's two entrenched errors was
"targeted supervised correction through `:learn`". **That was wrong**, and
checking it was the first thing this section did.

`:learn` cannot correct a routing error, for two independent reasons. Its
supervised path calls `router.learn(reply, category)` — it learns from the
*answer text*, not from the query that misrouted, so it cannot move a query it
was never shown. And none of its eight question kinds surfaces a misroute at
all; they are all knowledge or pattern gaps. Measured: answering a fact about
arithmetic moved it **1.2 → 1.4** while the wrong winner sat unmoved at **2.15**,
so the 0.95 gap would close 0.2 at a time and only when the answer's words
happened to overlap the query.

The only ground truth for a routing error is a person saying so, and there was
no way to say it. `CategoryRouter.correct` and the `:correct` command are that
way:

```
:correct arithmetic add these two numbers
```

It does **two** things, and the second is what matters against entrenchment: it
strengthens the query's own content tokens for the right category, *and weakens
whichever category is currently winning them*. Adding weight alone has to
out-climb an accumulation; removing the accumulation reverses the decision.

Two limits are deliberate. Base keywords are never unregistered — those were
chosen on purpose and one correction is not grounds for discarding vocabulary.
And learned weight never goes below zero, so a correction can remove evidence
but cannot manufacture the opposite claim.

**Applied to the deployed library:**

| | before | after |
|---|---:|---:|
| genuine routing | 0.750 | **1.000** |
| decoys declined | 1.000 | 1.000 |
| `crosses` on its own queries | — | 3.7 / 3.2 / 2.95, all still won |

Both corrections reported `weakened: crosses -0.55, frames -0.1` and landed
first-try.

**No gate module accompanies this, and that is a judgement rather than an
omission.** A gate is for a mechanism whose benefit is statistical and could be
mistaken for noise; a correction either reverses the decision it names or it does
not, so the claim is directly checkable. What genuinely needed a control was the
collateral — a correction must not damage the categories it takes weight from —
and `tests/test_route_correction.py` checks that the weakened category keeps its
own queries and that abstention does not reopen.

**One fixture mistake worth recording.** The first version of those tests built a
router where `crosses` lacked the keywords that made it win in the first place,
so nothing was entrenched and three tests failed — for the right reason. The
fixture now carries the deployed shape, because a test for entrenchment that
never entrenches proves nothing.

### 11.19 The storage split made real, and training one voice at a time

Two changes with one shape: the vault is the permanent store, and everything
short-term is byte-bounded RAM with disk reachable by reference.

**The context window is live.** §11.14 built and gated it; nothing used it. It
is now a `Session` field: every pipeline turn is appended (after the thoughts
run, so a turn can never match itself), `Recall` consults it, and a turn evicted
from the resident window is paged back through its 24-byte reference. Verified
in the wired pipeline, not just the gate: 51 turns at a 512 B budget left 6
resident, and "how deep is the harbour" recovered the buried fathoms turn —
the trace reads `2 turn(s) paged back from disk`. `:resident` now shows the
split in one place: expert cache budget, context window residency, and what
lies on disk behind both.

**Training runs one voice at a time, smallest first, until full.** The bulk
all-voices trainer was replaced on request, and each part of the shape carries
a reason. One at a time, because the teachers are 7–48B models on one card and
several resident was measured to thrash it. Smallest first, because the 7B
costs seconds per category where the 35B cost 65 s — the library banks most of
its coverage before an expensive model is ever loaded. And "until we have
enough" is made precise by the §11.13 ceiling: each category absorbs at most
12 distilled phrasings **ever, across all runs**, tracked in
`vault/distilled.json`. When every category is at budget the run reports the
catalogue full and teaches nothing.

Three defects were caught before they cost anything:

* **`min()` ordered a 19.6 GB model first.** MoE ids name both totals and
  actives (`qwen3.6-35b-a3b` is 35B total, 3B active), and taking the minimum
  put the 35B in front of the 7B. Load cost tracks the total; `max()` does.
* **The cap was per teacher.** Four voices at 12 each would have taught 48 per
  category — quadruple the measured ceiling, through the very module whose
  docstring warns about it. The cap is now enforced across teachers at the
  merge, round-robin so every voice contributes before any contributes twice.
* **The ledger over-counted by 2x.** It recorded every *generated* phrasing,
  but only the taught half ever touches the router — the held half is
  measurement. One teacher would have filled the whole ledger with half of
  every category's real budget unspent, and run 2 would have reported
  "catalogue full" after a single voice. It records taught phrasings only, and
  the existing ledger was repaired by recomputing the seeded split.

**The runs so far.** Run 1, `mistralrp-…-mistral-7b` (7B, smallest): 8
categories, held-out routing 0.562 -> 0.604   (+0.042), decoys held at 1.000. Run 2,
`katarau-9b-ru-rp-nsfw`: 24 phrasing(s), held-out 0.458 -> 0.500   (+0.042), decoys held.
Budget consumed per category: arithmetic:9, arrows:9, crosses:9, frames:9, geometry:9, language:9, lines:9, world:9 of 12.

The queue is voices, not models — a lineage sibling of a used teacher adds
near-identical phrasings (§11.12), so `cydonia-22b` never runs once
`mistral-7b` has, and the remaining queue is the 31B and the unsized
`command-r`, largest-last.

**Run 4 regressed the library, and turned the decoy check into a gate.** The
last voice, `command-r`, emitted the phrasing `Sign of the
times<|END_OF_TURN_TOKEN|>` — its chat-template token leaked into the
completion, survived `_clean`, and the router learned `end`, `turn` and `token`
as evidence for `crosses`, plus `times`, which plural-folds to `time` and
claimed "what time does the train leave". Held-out routing fell −0.125 and
decoys fell **1.000 → 0.800**. The trainer printed exactly that — and saved the
router anyway. The check was a caption where it needed to be a gate.

Three changes came out of one bad run:

* `_clean` strips `<|…|>` template tokens, so a teacher's chat scaffolding can
  never again become routing evidence.
* `CategoryRouter.unlearn` and `ShardVault.weaken` — the exact inverses of
  `learn` and `reinforce`, floored at zero. Learn-then-unlearn restores the
  prior table exactly, which a test asserts.
* `_teach_and_measure` **rolls back any run whose decoy score falls**: the
  teaching is unlearned in the same transaction, nothing reaches the ledger,
  and the report reads `ROLLED BACK` instead of a warning above a save.

Run 4's damage was then undone by the same mechanism applied retroactively
(1.60 weight removed), and the library measures **decoys 1.000, genuine 8/8**
with the budget back at 10 of 12 per category. `command-r` stays recorded as
used — re-running it would produce the same junk — so the voice queue is
exhausted: four voices taught, one rolled back, largest last, exactly the
sequence asked for.

### 11.109 Stage 1, in the domain it was told to try

§11.3 states Stage 1's gate before the work, so the answer cannot be
adjusted afterwards: *"a held-out category trained on 5 examples
through the encoder beats the same category trained on 5 raw-pixel
examples, by a margin larger than seed variance."* §11.5 ran it on
pixels, it FAILED, and it named where to go: raw pixels were already
near-ideal for strokes, so a re-encoding could only lose - and a
shared representation should pay in *"a perceptual domain where the
raw features are a poor representation"*. §11.108 established that
text is such a domain and that the advantage there is semantic.

So: Stage 1's criterion, unchanged, in the domain that failure
pointed at. **The only liberty is the translation**, stated rather
than slipped in: "raw pixels" becomes binary bag-of-words over the
corpus vocabulary, and "through the encoder" becomes the pretrained
embedding.

**Five examples means five, from a corpus written for something
else.** Every category has exactly six texts, so leave-one-out gives
5 train and 1 test with no sampling and nothing newly written.
Hand-writing extra paraphrases for a gate that rewards paraphrase
handling is how a deck gets stacked without anybody deciding to.

**PASSED**, six seeds, six folds:

| | real | permuted |
|---|---:|---:|
| bag-of-words | 0.296 | **0.296** |
| embedding | **0.463** | 0.301 |
| margin | **+0.167** | +0.005 |

Margin **+0.167 at seed sd 0.099**; difference **+0.162 at sd
0.130**. Both clear their bars and **both clear them thinly** - 1.7x
and 1.25x. Six categories, thirty-six test items per seed. Reading
those as large would be the error this project is arranged against.

**Two harness defects were caught by the criteria before the result
was believed, and they are worth more than the pass.**

Criterion 3 failed first: bag-of-words scored 0.296 real against
0.269 permuted, where a bijection guarantees equality. The permuted
vocabulary had been sorted independently, so feature index i meant a
different word in each condition and the two "identical" arms were
different arithmetic against the same seeded weights.

Before that, the vocabulary-size assertion **found a leak in
§11.108's control** - `permute` split on whitespace, so
"equal-sided" matched no key and passed through untranslated,
leaving real English meaning inside the condition that exists to have
none. §11.108's numbers are corrected there (+0.514 at 4.8x becomes
+0.472 at 3.7x; 92% semantic becomes 85%). **A gate found a defect in
the gate before it**, which is the only reason this one can be read.

**What this settles.** §11.2 named two things missing: composition,
solved at Stage 2, and shared representation, which failed at Stage 1
and was recorded as abandoned in that form. **A shared representation
does buy few-shot transfer, and the transfer is semantic rather than
capacity.**

**What it does not settle, and the distinction is exact.** Stage 1 is
*"a shared encoder... one representation over the feature space"* -
one this system **builds**. `nomic-embed-text-v1.5` is pretrained and
reached over the network. **The criterion is met with a borrowed
representation; the mechanism is not built.**

What changes is why that matters. Before this run it was an open
question whether a shared representation would help this system at
all - §11.5 said no on pixels, §11.11 could not read the answer on
text. It is now measured that it does. **Learning one is a
well-motivated engineering problem rather than a bet**, which is a
different and much better place to be stuck.

### 11.108 The edge was semantic after all

§11.2 named two things missing from this architecture: composition
and a shared representation. Composition landed at Stage 2. **Shared
representation had failed twice** - §11.5 on pixels, where raw
features were already near-ideal, and §11.11 on text, where the
effect was large and the control was not clean enough to read it.

§11.11 measured **paraphrase +0.533 at 5.66x seed sd**, reported FAIL
because its partial-overlap control also moved, wrote down the test
that would resolve it, and refused to use it:

> *"A difference-in-differences would read +0.446 and pass easily.
> That is probably the better-specified test... But it was formulated
> AFTER seeing the numbers, and adopting a criterion because it
> converts a fail into a pass is the goalpost move §11.5 refused."*

**So it was registered in advance, with a control built for it.**

**The control: a consistent bijection onto unrelated real words.**
Every content word becomes a different real English word, the same
way everywhere. What that does and does not disturb is the entire
design:

* **Lexical routing is mathematically unchanged** - token overlap is
  invariant under a bijection. So lexical accuracy must come out
  identical, and criterion 1 checks it. **A control that proves its
  own faithfulness** is what the partial-overlap control could never
  offer.
* **Capacity is unchanged** - real words in, well-formed 768-dim
  vectors out.
* **Meaning is destroyed** - "a box outline" and "a square shape"
  become unrelated sentences.

A capacity advantage survives that. A semantic one cannot.

**PASSED on all four criteria:**

| | real | permuted |
|---|---:|---:|
| lexical | 0.222 | **0.222** |
| embedding | **0.778** | 0.264 |
| margin | **+0.556** | +0.042 |

**Difference +0.514 at seed sd 0.107 - 4.8x the noise.** Lexical came
out identical across eight independent permutations, so the control
held exactly. **92% of the embedding's advantage disappears when the
meaning does.** And the permuted embedding still beats chance (0.264
against 0.167), so the control did not win by demolition - though
that margin is modest and should be read as modest.

**What this settles and what it does not.**

It settles §11.11: the effect was real, and the test that shows it is
the one that unit named and honourably declined to apply after the
fact.

**It does not pass Stage 1**, and the temptation to say so should be
named rather than resisted quietly. Stage 1's gate is **few-shot
transfer**; this is **routing**. What is established is the PREMISE
Stage 1 rests on - that a shared representation carries structure raw
features do not, and that the structure is semantic rather than
capacity. Whether it buys transfer is a different measurement.

**And the representation is not ours.** `nomic-embed-text-v1.5` is
pretrained and reached over the network. §11.5 and §11.11 tried to
LEARN a shared representation and did not succeed; this measures that
a good one exists and helps. A system whose shared representation is
an external service has bought transfer at the cost of the
from-scratch principle, and that trade is not made here - nothing in
the pipeline depends on this. It is a measurement, not a dependency.

Six categories, eighteen paraphrases, one embedding model.

### 11.107 Three defects, measured before they were fixed

Three things were known to be wrong and shipped anyway, each recorded
in an earlier unit and then left. None was a missing capability; all
three were things the record already said were broken, which makes
them worse than absences.

**The plateau rule only ever cost accuracy.** §11.104 measured
**11.8% of its plateaus as justified** - it abandoned answerable
questions nearly nine times in ten - and it stayed in the code. Swept
before removing it:

| stall reads | reads | accuracy | plateaus | justified |
|---:|---:|---:|---:|---:|
| 2 (as shipped) | 2.56 | 0.9741 | 2.6% | 14.3% |
| 3 | 2.59 | 0.9815 | 1.9% | 20.0% |
| 5 | 2.63 | **1.0000** | 0.0% | 100.0% |
| off | 2.63 | **1.0000** | 0.0% | 100.0% |

At every setting where it fires it fires **wrongly**, and at the
setting where it is justified it **never fires** - so "5" and "off"
are one policy written two ways, and the 100% is vacuous. It bought
2.7% fewer reads for 2.6 points of accuracy. **A rule that is only
ever wrong is deleted rather than tuned**, and the sweep is kept in
`shards/enough.py` so nobody adds it back by reasoning.

**The float spelling was worth asking about, and less than the
phrasing implied.** §11.98 measured 86.5 bits/weight for
exactly-stored floats, called it "JSON writing decimal digits, a
spelling problem rather than an information one", and parked it.
Measured on the same weights: JSON decimals **82.75**, packed float64
in base64 **66.91**, float32 **34.75**. float32 does not round-trip,
so it was refused - criterion 1 has no tolerance to widen. Shipping
the packed form took the shard from **86.500 to 72.531 bits/weight**,
and the gate re-ran with zero differing logits.

That is 16% through the whole shard, not the order of magnitude
"spelling problem" suggested. The remaining cost is float64 itself,
which is information rather than spelling, and the note now says so.
The old `"floats"` payload is still read, because a stored shard
outlives the code that wrote it.

**The flaky test was not reproduced, and what was found instead was
a leak.** One failure in roughly ten full-suite runs, never seen
again. Every test seeds its randomness, and the network tests are
mocked - the only real sockets in the suite are threaded
`ThreadingHTTPServer` instances in three modules, and **four of their
teardowns called `shutdown()` without `server_close()`**, which is
what the ResourceWarnings in every run had been reporting.
`test_codeweb.py` already did it correctly; four sites did not. Fixed,
and those modules now run clean under `-W error::ResourceWarning`.

**Whether that was the flake is unproven and is not claimed.** Leaked
descriptors accumulating across a two-thousand-test suite is a
genuine flake mechanism and this was a real defect worth fixing on
its own terms, but the failure was seen once and never reproduced, so
the honest statement is that a leak was found and closed while
looking for something else.

### 11.106 What it costs to change your mind

§12.3 named three axes where these architectures differ most
structurally and where nothing had been measured: **correction
latency, resident bytes per fact, and whether an answer can be vetoed
premise by premise.** This measures all three. **Two came back ties,
and the most useful thing in the unit is a measurement error it
caught in its own favour.**

**Run one reported the transformer collapsing** - 100%, 60%, 30%,
**20%** as corrections accumulated. That would have been a
spectacular result for this architecture and was entirely an artifact
of the harness. The failures were not wrong answers, they were
**EMPTY** ones: `qwen3.8-27b` is a reasoning model, `complete` was
called without `reasoning_effort`, and an 80-token ceiling left it
thinking until the budget ran out. `interpreter/llmls.py` had already
documented this exact failure - *"gemma spent all 400 tokens thinking
on every panel question"* - and passes `NO_REASONING` for it. This
gate did not.

**That is the second time an instrument error pointed the flattering
way**, after §11.105's empty-answer bug. When a comparison favours
the thing you built, check the instrument before believing the
result. Empties are counted separately now.

**Run two, corrected:**

| corrections | UQ accuracy | LLM accuracy | UQ bytes/query | LLM bytes/query |
|---:|---:|---:|---:|---:|
| 1 | 100% | 100% | 25 | 125 |
| 5 | 100% | 100% | 25 | 434 |
| 10 | 100% | 100% | 26 | 839 |
| 20 | 100% | 100% | 26 | **1691** |

**Correction accuracy: a tie.** A 27B model tracks twenty
accumulating in-context corrections without a miss.

**Auditability: a tie.** Asked which facts it used, the transformer
lists them. UltraQuant names its premises unasked, which is a
difference in default rather than in capability.

**Bytes per fact: the one real difference, and it is asymptotic.**
The context grew **13.5x across a 20x increase in corrections** -
linear, about 82 bytes each - while the store stayed flat at 25 to 26.
The difference is structural and genuine. It is also worth 1.7 KB at
twenty corrections, which a context window measured in tens of
thousands of tokens does not notice. **The crossover is somewhere in
the thousands and this gate did not measure it**, so what is
established is the slope, not the wall.

**The premise veto worked for both.** Derive "the anvil hardness is
high" from two facts, correct one premise, re-derive: both returned
"low", both citing the corrected premise.

A thinner result than §12.3 implied was waiting, and it is the
result.

### 11.105 UltraQuant against a transformer, on the same facts

ARCHITECTURE.md §12.3 said this repository had never compared
UltraQuant to a transformer, and that the strongest claim it made was
therefore the one with no gate behind it. This is that gate, and
**the claim it was built to establish did not survive it.**

**The comparison was built to be hard to rig**, because the obvious
version is not: choosing tasks this system was designed for and
reporting a win would measure only the choosing. Three guards. The
identical nine facts go to both, word for word, in the transformer's
prompt. A **world-knowledge family the transformer is expected to
win** is in the battery, and criterion 3 FAILS the gate if it does
not - a battery where the opponent never wins is not a battery.
And scoring is generous to free text: correct if the expected value
appears anywhere, refused if any recognisable refusal phrase does.

Against **qwen/qwen3.8-27b**:

| family | UltraQuant | transformer |
|---|---:|---:|
| held facts | 100% | 100% |
| derived (two-fact chains) | **100%** | 67% |
| arithmetic | 100% | 100% |
| world knowledge | 0% | **100%** |
| **fabrication on unheld subjects** | **0%** | **0%** |

**The transformer fabricated nothing.** Asked about five subjects the
facts never mention, it declined all five - and declined them again
when the run was repeated with the permission to decline REMOVED from
its system message (4 refused, 1 empty, 0 asserted).

**So §12.1's refusal claim was wrong, and it was wrong by conflating
two levels.** "A softmax always emits; abstention is not expressible
in its output shape" is true of a **classifier head** - that is what
§11.102 measured, and an untrained network reporting 0.184 of 3.00
bits is a real property a bare argmax cannot give you. It is not true
of a **language model**, whose output is text, and for which "I do not
know" is an ordinary sentence. §12.1 now says the narrower thing.

**A detector bug was caught, pointing the flattering way.** Run one
classified answers as refused or asserted, with no third case - so an
EMPTY answer, which one question produced, was scored as a
confabulation that had not happened. That inflated the transformer's
fabrication rate by 20 points on a five-question family, in the
direction favouring this system. `classify` now returns three
outcomes and criterion 4 covers all of them. **An instrument whose
error flatters its owner is the one to check first.**

**What UltraQuant won was not the claim under test.** On two-fact
chains - "the tower material is steel", "steel hardness is high", so
what is the tower hardness? - it scored **100% against 67%**. Three
questions, so an observation rather than a result.

**Limits, severe and to be read with the table.** One model, one
setting, twenty-four questions. The arithmetic family did not
discriminate at all: the cases chosen are ones a 27B model handles,
and a battery separating exact rational arithmetic from floating
point would have to be built deliberately - picking pathological
cases to make the point would be exactly the rigging the design
guards against. The world-knowledge family exists to keep the
comparison honest, and UltraQuant scores zero on it by construction:
a real limitation of the architecture, not an artifact of the test.

### 11.104 Reading until you know enough (failed, kept)

The thesis this whole library rests on, finally put behind a gate: a
dense model spends its entire parameter space on every token, and it
should not have to. Asked "what is 2 + 2" a transformer does the
same billions of multiply-accumulates it does for a question at the
edge of its competence, because a dense forward pass has no way to
do less. **Cost is a property of the model, not of the question.**

The library has had the routing and the paging for a long time. What
it never had was the STOPPING - `predict` then `prefetch` is one
shot, and whether its guess was too much or nowhere near enough, it
reads exactly that. `shards/enough.py` is the loop: experts
consulted in router order, and after each one the answer's own
evidence in bits decides whether to spend another.

**FAILED - and the two halves of the result point opposite ways.**

**The sharding half is vindicated, and it is the bigger claim.**
Reading a fixed **3 of 12** experts scores **0.9900** - identical to
reading all twelve.

| experts read | 1 | 2 | **3** | 4 | 12 |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.9450 | 0.9733 | **0.9900** | 0.9917 | 0.9900 |

**A quarter of the library buys all of the accuracy.** That is the
claim a dense model structurally cannot make, and it is now measured
rather than asserted.

**The stopping half is refuted.** Adaptive recall was beaten by a
plain constant, three separate ways: against `round(mean reads)`
(0.9667 vs 0.9833), across a sweep of twelve settings of router
noise and evidence floor (**lost at 11 of 12**), and against the
fixed-k curve interpolated at adaptive's own 2.54 reads (0.9667 vs
**0.9823**). The third is the one that counts, because run two's
comparison had been unfair in adaptive's FAVOUR - `round(2.54) = 3`
gave the constant a bigger read budget than the thing it was
competing with.

**Run one failed differently and the fix did not rescue it.** The
loop first returned the first expert to clear the floor. The sweep
showed why that loses: first-past-the-post against best-of-k is not
a contest about cost, it is a contest about **comparison**, and
stopping at the first adequate answer throws away exactly what the
constant was quietly buying. Run two kept the best and confirmed
nothing better followed - it read twice as much (1.25 to 2.54) and
still lost.

Criterion 1 says the same thing from the other side: reads varied
with sd 0.64 then 0.76, under the one-read bar. **The cost is very
nearly constant, so the loop is an expensive way of computing a
number that could have been a parameter.**

**Criterion 5 failed outright and found a real bug.** Of recalls
that stopped on "plateau", only **11.8%** were questions the full
scan could not answer either - the rule abandons answerable
questions almost nine times in ten. That is a bug wearing a policy's
clothes, which is what the criterion was written to catch.

**The limitation, stated as a limitation and not as a defence.** In
this world every question has the same shape, so questions do not
differ in how much reading they need, and adaptive stopping can only
pay where that varies. Building a world with heterogeneous
difficulty and reporting THAT would be searching for a setting that
passes - the move the sweep exists to refuse. The honest statement:
**on questions of uniform difficulty a constant k beats
entropy-driven stopping, and the condition under which stopping
might pay is untested here.**

### 11.103 An encoder that refuses its own output

§11.101 measured what ternary conversion does to a 27-block tower:
the projected output ends at cosine 0.0185 to the f16 original - not
degraded, uncorrelated. **And the tower did not notice.** Every
activation stayed finite, every shape stayed right, and it emitted
plausible numbers all the way through. The only thing that knew was
the f16 reference, which in a real deployment does not exist.

§11.102 solved that for the prediction layer, where the output IS a
distribution. An encoder's output is not - it is a list of real
vectors, and there is no softmax to take the entropy of.

**But the attention rows are distributions, and they are the only
genuine ones inside a transformer.** Each is a softmax over
positions; its entropy, normalised by `log2(seq)` so 0 is perfect
focus and 1 is attending to everything equally, says how much
structure the tower found - from the forward pass alone, with
nothing to compare against. `infer/confidence.py` fits a band on
healthy runs ONLY, because that is the deployment situation: you
have the model you shipped and no oracle.

**Run one failed on the detector rather than on the measure.** The
held-out healthy tower was REFUSED. The band had been fitted on two
runs whose entropies sat 0.017 apart, with a width proportional to
that observed spread - so the band was 0.017 wide. **A band fitted
on two runs is a band the width of the gap between them**, which is
a detector that has learned the two runs it saw rather than what
healthy looks like. `fit_band` now refuses fewer than three traces
outright, uses mean +/- 4 sd, and floors the assumed spread at 2% of
the mean.

**Run two, five images, four fitted and one held out: PASS.** Mean
attention entropy 0.6170 healthy against **0.9787** broken; the
held-out healthy tower accepted, the ternary tower refused.

**Criterion 4 is where the unit earned its keep**, because it forced
each measure to be judged alone:

| measure alone | accepts healthy | refuses broken |
|---|:--|:--|
| attention entropy | yes | **yes** |
| peak residual RMS | yes | **NO** |

**Attention entropy does all the work; peak residual RMS contributes
nothing.** In run one RMS appeared to catch the broken tower, but
that was the too-tight band - given a properly fitted one, 53.969
sits comfortably inside the healthy range and the ternary tower is
ACCEPTED on that measure. Peak RMS is the closest thing here to what
production monitoring actually watches, and on a tower reduced to
cosine 0.0185 it notices nothing. The non-finite check notices
nothing either. Both stay in the conjunction, because they may catch
failures this one did not exercise, but the record says which fired.

**The per-block profile is why it works, and it is legible.** The
healthy tower's attention entropy rises to 0.91 in the first blocks,
falls to about 0.30 by block 21 as the tower learns what to look at,
then broadens again. The ternary tower starts at 0.83 and climbs
monotonically to **0.997**:

| block | 0 | 5 | 10 | 21 | 26 |
|---|---:|---:|---:|---:|---:|
| healthy | 0.65 | 0.76 | 0.64 | **0.30** | 0.66 |
| ternary | 0.83 | 0.98 | 0.98 | **0.99** | 1.00 |

**The converted tower never becomes selective.** At 0.997 of maximum
entropy it attends to every position equally - which is to say
attends to nothing - and that is a far more legible account of what
ternary conversion did than "cosine 0.0185" was.

### 11.102 A prediction layer that says how much it knows (failed, kept)

`UltraQuantNet.predict` returned the argmax and its softmax
probability. That is the standard thing to report and the standard
thing to be misled by: an argmax always exists, so a network that
has learned nothing still names a class and still puts a number
beside it. Shown something that is not a glyph at all, the
recognizer answered "stripes_h, 0.31" and nothing in the reply said
the question was never answerable.

That broke a line the rest of the system keeps everywhere else.
§11.48 refuses to answer "is X Y?" about a subject nobody holds; the
pipeline says "I don't know" in a dozen places. The prediction layer
was the one component that could not, because its output shape had
no room for it. `model/uncertainty.py` gives it room: entropy in
BITS, the evidence `log2(n) - H` that grows with knowledge rather
than shrinking with it, the margin to the runner-up, and floors
below which the layer declines.

**FAILED criterion 2: entropy is the WORST of the three measures at
the job it was proposed for.** Ranking the same predictions and
discarding the least certain fifth - margin 0.0078, max-softmax
0.0130, entropy **0.0260**. Entropy loses to the plain baseline it
was meant to improve on, and loses again to `margin`, which came
from the same new module. The reason is structural: **entropy is
moved by the whole tail**, so a confident, correct prediction with a
little probability spread across the other seven classes scores as
uncertain, while max-softmax and margin look only at the top, which
is what decides whether the argmax is right.

**Criteria 3, 4 and 5 passed, and they are what entropy is actually
for.** An untrained recognizer reports **0.184 of 3.00 bits** - it
says it knows nothing. Max-softmax on the same predictions reads
0.191, which means something only once the reader knows chance is
0.125 on eight classes, and that comparison is not in the number. A
floor of 1.774 bits fitted on one split keeps 79.8% of another at
**0.0178 error against 0.0750** for everything - declining the
least-certain fifth cuts errors 4.2x - and **98.8% of non-glyph
noise is declined against 20.2% of real glyphs.**

**So the measures divide by which failure they catch**, at matched
coverage:

| floor on | accepted error | noise declined |
|---|---:|---:|
| evidence | 0.0229 | **98.5%** |
| margin | **0.0077** | 93.8% |
| weaker of both | 0.0222 | 98.5% |

**`margin` catches wrong predictions; `evidence` catches inputs the
network has never seen.** A hard in-distribution case is a two-way
confusion - small margin, respectable evidence. An
out-of-distribution input spreads everywhere - low evidence, and a
margin that can still be accidentally wide. Both floors ship,
separately, because neither is the other.

**Requiring both was tried and does not compose**: taking the weaker
of the two lands at evidence's error rate with none of margin's
gain, because whichever measure ranks a sample lower dominates, and
that is usually evidence. Recorded so it is not retried.

The comparison table was run AFTER the criteria were fixed and is
reported rather than gated. The registered question was "does
entropy beat max-softmax", the answer is no, and dressing the
follow-up up as a pass would be scoring the coin after the throw.
The unit ships anyway, because a failed criterion produced a better
design than the one it was testing.

### 11.101 The 27-block encoder

`infer/vit.py` assembles the real vision tower from the real
checkpoint: 27 blocks, 1152 wide, 16 heads of 72, a 4304 feed-
forward, a 48x48 patch grid, and the `qwen3vl_merger` projector that
concatenates 2x2 neighbouring patches before two 4608-wide layers.

**Every hyperparameter is read from the file, and one of them proves
why.** The checkpoint specifies `layer_norm_epsilon = 1e-6`;
`ops.layer_norm` defaults to `1e-5`. Trusting that default one unit
after building the machinery to catch exactly this class of error
would have introduced a silent, everywhere, plausible-looking
mistake. `load_config` raises on a missing key rather than
defaulting, and a test asserts that for every key in turn.

**Streaming, because the weights do not fit.** One block is 15.2M
weights; twenty-seven is 402M, which as Python lists is over ten
gigabytes. A block is read, used by every tower under test, and
dropped - the residual stream is the only thing that persists. The
gate walks six towers through one pass over an 857 MB file.

**A real bug, caught by a size check rather than by its output.**
`TensorInfo.rows` is `dims[1]`, which is the output count for a
matrix and is 16 for the 4-D patch kernel - so reading `info.rows`
rows returned 256 of 884,736 values. Without the assertion the
kernel would have been built from a fragment, every patch would have
been scrambled identically, and the encoder would have run to
completion producing finite, plausible, meaningless numbers.

**The tower runs, and its profile is a real one.** Residual RMS
dips through the early blocks (0.58 to 0.28), grows smoothly from
block 9 to 22 (0.48 to 2.00), and then spikes to **71.7 at the final
block** - the massive-activation signature real SigLIP-shaped towers
show. Nothing was fitted to produce that; it is what these weights
do.

**What the gate could not check, said plainly.** No reference
implementation on this machine exposes intermediate activations - LM
Studio ships `llama-server.exe`, which returns text. So pre-norm
ordering, the contiguous head split and the 2x2 merge are read from
tensor names, shapes and metadata, and they are ASSUMPTIONS. The
gate compensates by measuring whether it can detect deliberate
mis-wirings at all.

**FAILED criteria 2 and 3, and the failures are the result.** The
compounding of ternary error across a stack had been asserted three
times in this repository and never measured. Measured, with the f16
tower as its own reference:

| block | cos(f16, ternary) | f16 RMS | ternary RMS |
|---:|---:|---:|---:|
| 0 | 0.9496 | 0.58 | 0.62 |
| 5 | 0.4889 | 0.29 | 0.28 |
| 10 | 0.2681 | 0.50 | 0.27 |
| 20 | 0.2280 | 1.50 | 1.47 |
| 25 | **-0.0592** | 1.63 | 6.90 |
| 26 | 0.8749 | 71.73 | 53.32 |
| **after the projector** | **0.0185** | | |

**0.0185 against a bar of 0.90.** A converted encoder's output is
not a degraded version of the original - it is uncorrelated with it.
The half-life is about five blocks, and by block 25 the towers are
mildly ANTI-correlated. The ternary tower also destabilises in
magnitude: RMS 7.21 at block 24 against 1.96, which the cosine
alone hides.

**The methodological trap is worth more than the headline.** Block
26 reads **0.8749** and the projected output reads **0.0185**. The
final block's massive activations dominate both towers, so they look
aligned while their content does not agree; `post_ln` divides that
shared magnitude out and the disagreement appears. Anyone measuring
conversion damage on the last hidden state, before the final norm,
would have read 0.87 and called it a success.

**Criterion 3 failed because quantisation damaged the output MORE
than any structural bug.** Interleaved heads 0.0193, SiLU-for-GELU
0.1210, a dropped attention residual 0.2218 - all above ternary's
0.0185. A tower with no attention residual keeps twelve times more
signal than a correctly wired one whose weights were converted.

**That failure was localised rather than accepted.** The same
comparison at depths 1, 2, 4 and 8 discriminates every time - all
three defects land below the ternary tower - and only stops at 27.
So criterion 3's failure is a fact about quantisation, not about the
method, and it is criterion 2's fact seen from the other side. The
wiring is validated where the signal survives to validate it.

Two smaller findings. **A wrong activation is instantly fatal and a
wrong head split is not**: at one block SiLU-for-GELU already reads
-0.0055 while interleaved heads still reads 0.9482, becoming obvious
only by block 8. The bug that hides for eight blocks is the more
dangerous one. And the **wrong epsilon measured 1.0000** - no effect
at four decimals over 27 blocks. It was registered as reported
rather than gated because its size was unknown; the answer is that
this default would have been harmless, and reading from the file is
still right because the next checkpoint's may not be.

**The standing conclusion, now measured rather than extrapolated:
conversion without retraining does not produce a working encoder.**
§11.95 said so from one layer's 0.884 and was too kind.

### 11.100 The operations between the matmuls

`model/network.py` never needed these, because an MLP is linear
layers and a ReLU. A transformer is not. `infer/ops.py` adds what is
missing and nothing else: LayerNorm and RMSNorm, softmax, the two
GELUs and SiLU, the contiguous head split, and scaled dot-product
attention with a causal mask. Deliberately shared - a vision encoder
and a text decoder differ in tokenizer, cache, mask and position
encoding, and do NOT differ in what a LayerNorm is.

**The oracle is the interesting design choice.** Every operation
here has a textbook definition that a WRONG implementation still
satisfies loosely: a softmax with the wrong denominator still sums
to one, a LayerNorm dividing by n-1 still centres, a GELU using the
tanh form where the model trained with erf still looks like a GELU.
Each is a silent few-per-cent error, in every block, compounding
over thirty layers into nonsense while every shape stays correct.
Property tests do not see any of them. So each operation is computed
independently at **50 significant digits**, straight from its
definition, with no stabilisation and no float arithmetic - erf by
its own Taylor series, pi as a literal - and the shipped float
version has to agree.

**Run one failed four ways, two of them mine and two of them the
criteria's.**

*A real bug*: a fully masked attention row came out **NaN**, 24 rows
of it. `softmax` subtracts its maximum, and when every score is
`-inf` that is `-inf - -inf` before the zero-total guard can fire. A
NaN row propagates silently through a whole block, and the row still
had the right length and the right dtype - exactly what a property
test misses.

*A real harness bug*: criterion 2 requires the naive implementation
to actually BREAK, and SiLU's unbranched form only overflows past
|x| ~ 709 while the range tested was [-8, 8]. It broke 0 times in
60, so the criterion had demonstrated nothing and failed on that
ground. The by-product is worth keeping: on real activation ranges
that branch is never needed. It is insurance, not a hot path.

*A bad criterion*: per-element relative error measures float64, not
the code. GELU at x = -8 is -4.98e-15, reached by adding 1 to
erf(-5.66) = -0.99999999999999502; the cancellation leaves two
significant digits, so relative error is 1.8e-2 while absolute error
is 9.2e-17 - one ULP. **No implementation in this format can pass
it.** The measure is now error as a share of the output vector's own
RMS, which is what matters for values about to be added into a
residual stream.

*A bad criterion*: the deliberately adversarial mean-1e6 stream was
gated, and no format can pass that either - the mean's own ULP is
1.16e-10 against centred values of O(1). Now measured and reported
rather than gated.

**PASS on run two: worst gap 3.80e-15 against a 1e-12 tolerance;
naive forms broke 15, 15 and 60 times out of 60; zero causal leaks;
row sums within 1.11e-16 of one; zero masked rows non-zero; zero
head-split errors; attention at 21.9 M MAC/s in pure Python.**

The reported column earns its place. On a large DC offset every
operation is untouched except **LayerNorm, which loses five orders
of magnitude (9.81e-11)** - and that is not a curiosity, it is the
one op that subtracts a mean, so it is the only one that can care.
RMSNorm, which deliberately does not centre, is unaffected. Anyone
debugging a drifting residual stream at layer thirty should have
that number before they start.

One smaller correction: the two GELUs were documented as differing
"by up to 0.0003". Measured, it is **0.000473, at x = -2.698**. The
pin asserts the measured value now, after the first version asserted
a bound nobody had looked up and failed on a point where the gap
happens to be 9.8e-5.

### 11.99 A glyph for every imported tensor (failed, kept)

Every packed tensor now renders a 5x5 glyph from its own trits -
each cell the density of surviving weights in its block, set when
that block is denser than the tensor's own mean, so overall sparsity
drops out and structure stays. The system already had a recognizer
for exactly that shape, so the glyph is NAMED by it and the name is
written as an association: `convert/glyph.py` renders,
`pack_checkpoint(..., recognizer=...)` names, and asking the catalog
for "stripes_v" returns real Qwen tensors.

**It failed criterion 2, and it took three passes to find out how
badly.** The headline moved twice, both times because the harness
was measuring something other than what the criterion asked.

**Pass one rigged the comparison.** The library already has a
content signature - `shards/sketch.py`, 64 bits of sign-random
projection - and the gate was built to measure against it. But the
sketch was handed 64 truncated row means while the glyph saw the
whole sampled block, and the glyph duly won by 1.5x (+0.179 against
+0.117). Given the same 25 block densities, with the sketch holding
strictly more of them, the result reversed: 1.32 sds against 0.85. A
comparison against a standard option is only worth running if the
standard option is allowed to win.

**Pass two was confounded by dimensionality, and that is the real
finding.** 26 of 40 tensors are 1-D, and those 26 render only **six
distinct glyphs** - a bias has one row, so four of five row-blocks
are empty and it renders a bar. 1-D tensors sit ~0.03 apart, 2-D
tensors ~0.48, and kinds are almost perfectly
dimensionality-homogeneous, so "within kind" was mostly measuring
"within dimensionality". On the 2-D pool alone:

| 2-D tensors only | margin | sd | margin/sd |
|---|---:|---:|---:|
| glyph (25 bits) | +0.020 | 0.103 | **0.19** |
| sketch (64 bits) | **-0.001** | 0.021 | **-0.04** |

**Neither signature separates tensor kind.** The glyph reaches 0.19
of the one-sd bar; the sketch separates nothing at all. That is not
a glyph problem but a summary problem - block density over a ternary
matrix does not tell an attention projection from a feed-forward
one. The mixed-pool number is kept in the report so the confound
stays visible rather than being corrected away quietly.

What passed is worth having and is smaller than the ambition. Zero
drift on a second render and through JSON, zlib and sha256; zero
tensors unreachable by shape; and on the 2-D pool same-kind pairs
share a label 0.222 against 0.143 for any pair - a real lift over a
baseline that already absorbs the skew, across 18 pairs. So the
glyph keeps the job it does: rendered once, looked at, named,
retrieved by that name. **Nothing routes by glyph distance, and
nothing should route by sketch distance over these summaries
either** - the more useful half of the result, and the half only the
standard option could have supplied.

This is the third time 1-D tensors have contaminated a measurement
here. §11.95 read 2.41 relative error by pushing a vector through a
1xN dot product; §11.96 compared small tensors against a baseline
carrying no metadata; §11.99 measured dimensionality and called it
kind. In a real checkpoint most tensors are 1-D and they behave
nothing like the matrices, so any aggregate that mixes them is
measuring the mix.

### 11.98 Weights that come back out and still recognise

§11.96 proved tensors go IN. That is a claim about BYTES, and it
says nothing about whether a network built from those bytes computes
what the original computed - "the weights are imported" is not true
until something runs on them. `convert/load.py` closes the loop: a
shard becomes a `TernaryLinear`, a set of shards becomes an
`UltraQuantNet`, and the glyph recognizer is the instrument, because
it is the system's one existing consumer of ternary weights and
small enough that every logit can be compared rather than sampled.

**The bar is exactness, for §11.96's reason one level up.** The
conversion above this layer is already lossy at 0.467 per layer; if
the import were lossy too, the two would be indistinguishable and no
future measurement of the quantiser could be trusted.

**Run one FAILED, and found a real defect.** All 128 glyphs differed
in their logits, 26 in their label, and accuracy fell **0.900 ->
0.792**. One layer caused it: `UltraQuantNet`'s head is full
precision by default, and the packer pushed it through the trit
codec like everything else - a 0.151 error in the layer that
produces the logits. **Trits are how you store a ternary matrix, not
how you store a matrix.** A layer that is not ternary now travels as
exact floats and says so in the payload; a converted transformer is
ternary by construction and never takes that path. That is the same
rule the bias decision rests on, and finding it twice in one module
is the useful part: it is not "biases are special", it is that a
lossy codec may only carry what is already in its alphabet.

**A mis-aimed measurement was corrected on the way.** The idempotence
check first ran on the ORIGINAL layers, whose master weights are
full precision and were never expected to survive requantisation. It
reported 0/2, which reads as a refuted claim and was a question
asked of the wrong object. Idempotence is a property of what
ARRIVES - and there it holds, which is what lets an imported layer
be handed back still flagged quantized.

**PASS: 128 glyphs, zero differing logits, zero differing labels,
accuracy 0.9000 -> 0.9000, zero bias elements changed, zero
unreachable layers, 1/1 ternary layers idempotent.**

**The cost, which is not a flattering number.** 1,216 weights in
3,606 bytes is 23.7 bits/weight, and the blend hides the trade:
ternary weights cost **6.208** bits each, exactly-stored floats
**86.500**. The head is 21% of the weights and 77% of the bytes -
that is what exactness costs, paid deliberately and named rather
than averaged away. The ternary rate is itself 3.4x §11.96's 1.820,
for a reason that is not the codec: at 960 weights the per-shard
metadata is most of the shard, and §11.96's number came from
1,018,016 weights. Neither is optimised here, and 86.5 bits/weight
is plainly JSON writing decimal digits - a spelling problem, kept as
the next question rather than left as an achievement.

### 11.97 The yes/no family, natively

Four question forms ported at once, because they share one door and
one discipline and splitting them would gate a half-open door: "is X
Y?" against a held belief, "is X A or B?" where the disjuncts are
offered values, "is A taller than B?" with units and derived
operands, and "is <derivable subject> Y?", which derives its own
subject through the chain machinery before it can check anything.

**The discipline is the thing worth porting carefully: absence is
never No.** A subject the library holds nothing about gets no
verdict - it falls through to the hedge, because "no" is reserved
for actual contrary belief, held either as a different value or as a
stored denial. That line is easy to state and easy to break in a
port, because every way of breaking it produces a confident,
plausible, wrong sentence. So the corpus asks about subjects that
are not there, and both tiers have to decline in the same words.

**One known gap, provoked rather than avoided.** A comparison side
may be an expression over BELIEFS - "is the tower height * 2 taller
than the bridge length?" - which needs §11.78's derived-operand
reader, and that is not in this tier. Rather than choose a corpus
that never asks, the corpus asks freely and the gate CLASSIFIES
every difference.

**PASS: 1,624 turns across 24 conversations, zero unexplained
differences, zero intent differences, zero stores differing, zero
fabricated verdicts over 109 questions about unheld subjects**, with
74 of the 89 belief-operand turns counted as the named gap.
Coverage: 613 statements, 176 polar questions about held facts, 140
comparisons that must refuse aloud, 114 against a written quantity,
113 choices, 111 between two held numbers, 92 with a derived
subject, 67 over literal expressions.

**A pass on the first run is a claim about the gate as much as the
port, so the gate was checked against itself.** With the branch
disabled and the tier rebuilt, 80 of 180 turns differ. A gate that
cannot fail has measured nothing, and this one can - which is worth
the two minutes it costs to find out, on a unit where 450 lines of
someone else's semantics were reproduced in one pass.

Two numbers are quieter than the headline. **Zero fabricated
verdicts** is a criterion checked SEPARATELY from parity, on
purpose: two tiers agreeing on a confident wrong answer would sail
through a parity gate without either of them being right. And **74
of 89**, not 89 of 89 - fifteen belief-operand comparisons agreed
anyway, because a side the Python tier also could not read refuses
in the same words. The gap is narrower than the form that provokes
it.

The unit also closed a stale scope claim. `interpreter.hpp` still
listed superlatives, aggregates and chains as unported after §11.92
and §11.93 landed them; the new pin reads only the NOT-ported
sentence, so the header cannot go quietly out of date again.

### 11.96 Tensors that go into the library

The conversion kit's output, packed INTO the shard library as a
first-class citizen - catalogued, addressable, paged - rather than
as a pile of files beside it. `convert/pack.py` is the codec,
`convert/library.py` writes shards, and both of its decisions are
visible in the catalog.

**Granularity: one shard per tensor.** A matmul needs the whole
matrix, so splitting a tensor across shards would page every piece
for every use; keeping several tensors in one shard would page a
layer to read a bias. The tensor is the unit that gets used, so it
is the unit that gets stored.

**Address: the category is the layer, the keywords are the name.**
Those carry different halves, and it is worth being exact about
which. The keyword index finds a KIND - "attn out" scores every
category holding an attention output projection - and it cannot tell
layer 3 from layer 9, because a layer number is a digit and digits
are not keywords. Layer identity lives in the category; kind lives
in the index; the shard id names the tensor exactly.

**The bytes.** The vault serialises JSON and zlib-compresses it,
which is an awkward home for millions of trits. Four spellings,
measured on real tensors: list of ints 2.25 bits/weight, byte per
trit 2.27, two-bit packed 1.53, base-243 packed 1.54. The dense
packings beat the obvious one by a third and then tie each other -
which the arithmetic does not predict, since five trits in a byte is
1.60 bits against two-bit's 2.00. The advantage vanishes in the
zlib: the looser packing keeps byte-aligned structure a compressor
can find, the denser one looks like noise, and density and
compressibility very nearly cancel. **base-243 won on the other
axis** - a table-driven decoder reads 118M weights/s against
two-bit's 34M, and a pageable library decodes on every page-in.

**PASS: 20 tensors, 1,018,016 weights, 1.820 bits/weight against the
naive spelling's 2.298 (+0.477 at 0.300 tensor sd), zero trit
differences, zero scale differences, zero unreachable shards, 11.6M
weights/s end to end.** Criterion 1 had no tolerance and held: the
conversion above this layer is lossy by 47%, and if this layer were
lossy too the two would be indistinguishable.

**A finding was retracted on the way.** Run one appeared to show
packing LOSING 0.4-0.6 bits/weight on 1-D tensors, and produced a
tidy story about skewed distributions needing a per-tensor spelling
choice. The baseline was unfair - a bare `{q, s}` dict with no name
and no shape against a payload carrying both, so it measured the
spelling and the metadata at once, and on small tensors the metadata
is most of the difference. Compared like for like, packing wins on
**twelve of fourteen** tensors and the two exceptions favour text by
0.014 and 0.062. The per-tensor choice survives for a much smaller
reason than the one it was built for: about 0.005 bits/weight, kept
because it can never be worse rather than because it pays.

### 11.95 What converting a transformer costs (failed, kept)

A kit that reads a trained transformer and puts it into UltraQuant's
ternary tier. `convert/gguf.py` reads GGUF with the standard library
- metadata, tensor table, and F32/F16/BF16/Q8_0 data - and refuses
every other quantisation BY NAME rather than guessing at its blocks;
`convert/ternary.py` quantises and, more importantly, measures;
`python -m ultraquant.convert <model.gguf>` prints what conversion
would cost that file.

**The measure is output error, not weight error.** A layer's job is
`W x`, so the number that matters is how far the ternary layer's
output lands from the trained one for Gaussian probes. Errors that
cancel across a row never reach the output.

Measured on 24 real weight tensors from an 899 MB checkpoint:

| rule | output error | cosine | non-zero |
|---|---:|---:|---:|
| per matrix, 0.75 (UQ's own) | 0.4942 | 0.8675 | 0.520 |
| per row, 0.75 | 0.4801 | 0.8753 | 0.529 |
| per row, optimal | 0.4741 | 0.8793 | 0.490 |
| chosen per tensor | **0.4667** | **0.8837** | 0.503 |

**Criterion 1 fails**: +0.027 against a tensor-to-tensor sd of
0.045. The rule choice is second-order; the loss is first-order.
Criterion 2 exists so the kit cannot hide behind a relative
improvement, so: **0.4667 relative output error at cosine 0.8837,
per layer.** A stack of thirty such layers does not preserve much.
Conversion is a starting point for retraining, not a substitute for
it, and the CLI says so in those words.

**Two things were learned that are worth more than a pass.** The
threshold that provably minimises the squared error, found in closed
form, produced a WORSE matrix-vector product than the 0.75 heuristic
on **7 of 24 tensors** - it keeps fewer, larger weights, and the
small ones it discards were carrying signal that cancelled less than
the Frobenius norm suggested. Minimising weight error is not
minimising output error. So the kit chooses per tensor, on one set
of probes and scored on another, because choosing and reporting on
the same probes is picking the winner after seeing the coin.

And a harness error caught before it was believed: the first pass at
1-D tensors reported **2.41** relative error and an obvious
conclusion ("biases must stay float"). It was pushing a random
vector through a 1xN dot product - measuring a bias as though it
were a matrix. Measured as what it is, a 1-D tensor takes **0.527**,
in line with the matrices. The clean conclusion was an artifact of
the wrong question.

### 11.94 One copy of a rule that has to agree

Housekeeping with a safety claim behind it. The stopword list and
the plural fold had been written three times in the native tier -
once each in the store, the spread and the interpreter - because
each piece was ported on its own and each needed them.

Three copies of THOSE rules is not a tidiness problem. §11.29's
coverage rule asks whether a question's INFORMATIVE tokens are
covered by a key, and the fold decides whether "towers" reaches
"tower". A copy that drifted would make the tiers disagree about
what a question asked - silently, in one branch only, which is the
hardest kind of difference to find and exactly the kind these gates
exist to prevent.

`native/uq/src/text.cpp` holds one copy now. **210 lines of
duplication removed, 27 added**, and all six native gates re-run
afterwards still pass - a refactor that changes behaviour is not a
refactor, and that is checkable here rather than assumed.

The pin is the point of the exercise. `tests/test_nativetext.py`
parses the C++ list out of the source and compares it to Python's
`_STOPWORDS` as a set, checks the length floor is carried too, and
asserts `normalize_token` is defined in exactly one file. "Copied
rather than reinvented" was a comment in three places; it is a test
in one.

### 11.93 A claim about a set, twice

Superlatives and aggregates, ported. Both stand on the same move -
enumerate the WHOLE store for facts whose key ends in an attribute -
and both carry the same honesty contract: scope named, exclusions
named BY NAME, ties named rather than broken, and denials never
counted in arithmetic, because a denial names no number and a denied
900 meters must not move a mean by a millimetre. Those trailing
clauses are why the gate compares sentences rather than winners.

**Run one found the best thing in this unit.** Two of 540 answers
differed, and only in the last digit:

```
what is the average weight?
  python: The average of the 6 weight facts I hold is 250.463 tonnes
  native: The average of the 6 weight facts I hold is 250.462 tonnes
```

Same facts, same order, same conversions, same formula. The cause is
not arithmetic but the SUM: **CPython's `sum()` over floats has
carried a Neumaier compensation term since 3.12**, so that family
sums to 1502.775 where a plain left-to-right accumulation gives
1502.7749999999999, and `%g` prints those as 250.463 and 250.462.
The native tier sums the way the oracle sums now - and the
alternative was refused deliberately: widening the comparison to
"close enough" would have turned this gate into one that cannot see
the next real difference either.

Two other details were ported rather than improved. The ranking
multiplies by the FLOAT unit factors because the Python tier does -
using the exact rationals there could order two facts differently at
the last bit and be wrong for a reason nobody could see. And the
combined value prints through `%g`, which is exactly what Python's
`format(x, 'g')` is.

**Run two: 540 set-claims across sixty worlds, zero answers
differed**, with the hard cases present and counted - empty families
and polysemous words in all 60, mixed units in 45, denials in 26,
non-numeric values in 24, ties in 21, uncomparable units in 16.

### 11.92 Two spreads that converge the same way

Spreading-activation inference, ported to the native tier and gated
against Python. The hardest piece to port faithfully, because its
correctness lives in ORDER as much as in arithmetic: which facts the
index surfaces for which probe, which lineage reaches a node first,
which of several equally-strong convergences wins. Python dicts
preserve insertion order and the spread leans on that, so the port
uses insertion-ordered containers wherever the Python side walks a
dict - a `std::map` would have sorted them and quietly changed which
chain answered. The activation strengths are powers of one half,
exact in binary floating point, so the one place a port would
normally drift, it cannot.

Everything else that could differ is a rule rather than an
implementation detail, and the rules were copied rather than
re-derived: whole-value bridging, the single-origin lineage, origin
coverage at one absent token, the terminal relation slot, the
adjacent-bigram order rule, and the anagram refusal.

**PASS on the first run: 368 questions across forty worlds, zero
derivations differed, zero curiosity gaps differed** - 130 two-fact
chains, 59 unreachable questions, 36 epithet decoys, 30 three-fact
chains, 30 four-fact chains, 30 denial bridges, 28 anagram decoys,
25 modifier questions.

The refusal-symmetry line matters most and held in both directions:
**zero questions were derived only by the native tier.** Every
epithet decoy, anagram subject and denial bridge that Python
refuses, the native tier refuses too. A port that converged more
often would have scored better on correctness and been strictly
worse, because each extra answer is a pun the rule stack exists to
refuse.

`missing_premise` came with it, which closed §11.91's debt: the
curiosity hint the native pipeline could not produce is produced
now, the chat gate's 71 hint-only differences went to zero, and
nothing else moved.

### 11.91 One conversation, two tiers, the same words

The native tier stops being a library here and starts being the
system: text in, a sentence out, a store that changed. The parity
obligation is at its strictest, because what a pipeline produces IS
a sentence — so the gate compares words, not a structure that could
be worded two ways, and compares them TURN BY TURN over a shared
conversation, because a store carries state and the third answer
depends on the first two.

**Ported here**: intent (statement, question, expression, chat);
statements parsed, stored and spoken back, with polarity, with a
revision named aloud, with §11.63's near-key adjacency note; and the
question ladder as far as the arithmetic and list readers, the exact
recall branch under §11.29's coverage rule, nearest-held, and the
honest unknown. **Not ported, and so not claimed**: polar, why,
comparatives, superlatives, aggregates, history, choices, testimony,
conjunctions, clause splitting, chains, glyphs, goals, URLs and
learning mode.

**One known gap is measured rather than avoided.** The Python tier
appends a curiosity hint to a hedged answer when a failed question
points somewhere — "... If I knew the iron bridge, I could work this
out — ':learn' will ask" — and that hint comes from
`missing_premise`, which belongs to the spreading-activation
machinery. Choosing a corpus that never triggers it would be a gate
quietly avoiding what the tier cannot do, which measures nothing. So
the corpus provokes it freely and every difference is CLASSIFIED: a
response differing only by that suffix is counted as a named gap, a
response differing in any other way is a fail.

**PASS on the first run: 1,200 turns across twenty conversations —
zero unexplained differences, zero intent differences, zero
conversations ending with unequal stores, and 71 turns differing
only by the curiosity hint.** Coverage: 420 statements, 193
questions about held facts, 150 arithmetic questions, 126 revisions,
118 questions that may or may not be held, 100 about nothing held,
93 lines of chat.

Two of those zeros are quieter than the headline and matter as much.
**Zero intent differences** means every turn went through the same
door in both tiers, which is what keeps the next turn comparable — a
right answer reached the wrong way diverges later, not now. **Zero
stores differing at the end** means the pipeline that said the right
thing also remembered the right thing. The 71 is the debt the
inference unit inherits, named and bounded rather than hidden.

### 11.90 Two stores that remember the same way

The native tier's fact store, gated against Python's. Facts are the
substrate every question form stands on, so parity here is wider
than "the same value comes back": a store is observed through
whether a statement was new, reinforced or a revision; what the
replaced belief was in spoken form with its polarity; what
confidence a fact carries after being told twice; which conclusions
a revised premise took down; and which key retrieval picks when
several overlap.

**What is deliberately not ported.** The Python store can put facts
in the pageable shard library; the native store keeps them in a map.
That is a storage decision rather than a semantic one — §11.86
established that the price at density is the bucket cache rather
than the lookup — so the gate compares against the store's plain
mode, which is the mode that defines what a fact IS.

**It took five runs, and every failure was the native tier being
reasonable instead of being faithful.** That is the finding, and it
is worth more than the pass, because none of the five was a wrong
value and each would have survived a gate that only checked what a
fact recalls to:

1. **Consolidation went through the statement path** — the obvious
   reuse, which logs a revision episode Python never logs and keeps
   a reinforcement count Python resets. A consolidation is not a
   statement.
2. **Retraction order** — Python walks a stack and appends each
   casualty as it finds it; the port swept and sorted. Same facts,
   different order, and the retracted list is spoken aloud when a
   belief changes.
3. **Episode order** — `recall_episodes` returns most recent FIRST,
   because a history answer reads the newest change first. The port
   returned insertion order: the same information in the wrong
   sentence.
4. **The plural fold in retrieval** — the port folded "towers" onto
   "tower" as the router does, but the store's own retrieval
   compares RAW tokens. A native tier that folded here would find
   facts the Python store does not, which is the same kind of wrong
   as missing them.
5. **`confirm_fact` raised rather than set** — being told "yes, that
   is correct" SETS confidence and can lower one, and it bumps the
   reinforcement count. The port took a maximum and skipped the
   bump. That count is exactly what §11.44's retrieval tie-break
   reads, so the mistake surfaced two steps later **not as a wrong
   confidence but as a differently ordered `find_facts`** — which is
   why the gate scripts whole sessions instead of checking calls one
   at a time.

**Final run: 4,800 script steps across twelve sessions, zero records
differed** — 2,502 statements, 461 recalls, 439 retrievals, 431
consolidations, 403 confirmations, 286 key enumerations, 278 episode
reads.

### 11.89 The same sentence, from a different language

§11.88 gave the native tier integers that agree with Python's. This
rung asks the harder question: does the whole arithmetic READER
agree — tokeniser, grammar, unit table, rounding rules, refusal
sentences, and the exact spelling of every answer?
`native/uq/src/calculate.cpp` is the port; the gate
(`experiments/nativecalc_gate.py`) is the reason to believe it.

**Parity is checked as a RECORD, not as a value.** Two tiers that
both answer "4" have proved very little. These have to agree on the
expression echo (with the precedence visible), the rendered value
("0.3" and "1/3", never a decimal that would be a different number),
whether a rounding happened and what the exact value was before it,
the bounds of an irrational root, and the WORDING of every refusal.
And on the quiet line that matters most: **a native tier that
answers more than Python is exactly as wrong as one that answers
less**, so every input Python declines must be declined natively
too.

**Run one caught exactly what the gate exists for.** Eleven records
out of 3,000 differed, all the same shape, none of them a wrong
number:

```
[list] 'list the maximum of 22 meters and 51 kilograms'
    python: refuse|no definition I hold connects meters and kilograms
    native: refuse|no definition I hold connects kilograms and meters
```

Both refused. Both refused for the right reason. They named the
units in opposite ORDER — because the Python tier computes the
running total before branching on the operation, so a mixed-unit
list refuses in the order the failing ADDITION met the units, not
the order a comparison would. A gate comparing values would have
called it a pass; a gate asking "did both refuse" would have called
it a pass. It is not one: the wording of a refusal is what this
system hands the person asking, and two tiers handing over different
sentences have not reproduced each other.

**Run two: 3,000 questions, zero records differed**, with coverage
reported so a green run cannot come from a corpus that quietly
stopped testing something — literal 506, rounding 430, list 415,
root 393, not-arithmetic 376, power 338, percent 291, spoken
operator 251 — and zero differences again at three further seeds.
The 376 non-arithmetic inputs ("what is the tower height?", "hello
there", "is 2 + 2 = 4?", "what is 5?") were declined by both tiers
every time.

One implementation note worth keeping, because it cost a debugging
pass: the C++ regex literals for the spoken-power rewrites survived
being written as raw strings, while two ordinary string literals in
the list reader lost their doubled backslashes in transit and
compiled to `\s` (one backslash) — a pattern that matches the letter "s". MSVC said
so (`warning C4129`), the list path went silently dead, and the fix
was to make every regex in the file a raw literal so the escaping
question cannot be asked twice.

### 11.88 A native tier that has to agree

README line 8 has stated the rule since the beginning: the
pure-Python tier defines this system's semantics, and every other
tier reproduces it. The existing `native/` sources are accelerants
for numeric kernels. `native/uq/` is a different thing — a native
build of the COGNITIVE core, C++17, no external dependencies, the
same rule the Python tier keeps by using no numpy.

The discipline is the point, and it is stricter than "port it".
Every piece is gated against Python as the oracle, not against a
second opinion of my own, and parity is checked as STRINGS: the
native tier is right when it says the same thing, not when it says
something defensible. Nothing here is required at runtime — a
machine with no compiler runs the whole system in Python and sees a
green suite, because the parity pins skip when the binaries are
absent. What must never exist is a native tier that is present and
wrong.

**The first piece is arbitrary-precision integers**, because C++ has
none and this system leans on Python's everywhere it matters: "what
is 2 ^ 1000?" is a 302-digit answer, an exact rational's denominator
grows without being asked, and §11.81's root enclosures scale a
radicand by 10^(places*degree) before taking an integer root. A tier
built on `long long` would diverge on the first interesting question
and be quietly wrong after that, which is worse than being slower.
`native/uq/src/bigint.cpp` is sign-magnitude, base 1,000,000,000,
limbs least-significant first — base 1e9 rather than 2^32 because
every decimal boundary in this system (parsing "0.1" exactly,
printing an exact decimal, scaling by a power of ten to round)
becomes limb arithmetic instead of repeated division, and those are
the paths that have to be exactly right rather than merely fast.

Two behaviours were implemented explicitly rather than inherited,
because they are where a C++ port disagrees silently. **Floor
division**: Python's `//` rounds toward negative infinity and `%`
takes the divisor's sign, while C++ truncates toward zero — and
every rounding rule in §11.84 is specified in Python's terms.
**Integer roots**: §11.81's enclosures are only honest if the root
is exact, so it is found by bisection over integers with no float
anywhere near it.

The gate (`experiments/nativebigint_gate.py`) **PASSED on the first
run: 4,000 operations compared against Python, zero mismatches,
0.494 of operands past 2^64, zero roots failing their own
exponentiation** — with add, sub, mul, floor-div, floor-mod, gcd,
pow, iroot and decimal printing each measured separately and each at
zero.

One difference between the tiers surfaced that has nothing to do
with arithmetic: **CPython refuses to print an integer past 4,300
digits** by default, a denial-of-service guard added in 3.11, and
the first attempt to build the case list crashed on it while
computing an ORACLE answer rather than while checking one. The
native tier has no such guard. The limit is raised inside the gate
so the oracle can speak at the sizes the tier under test can reach —
otherwise the measurement would quietly have become one of CPython's
printing policy.

### 11.87 A list of numbers is not a list of beliefs

"what is the average of 3, 5 and 10?" is basic arithmetic and the
system could not read it. §11.57's aggregate owns those exact words
over HELD facts — "what is the average height?" ranks the store —
and a written list looks identical from a distance, which is why
this rung is mostly a question about a boundary rather than about
arithmetic.

The boundary is drawn where it can be checked: **every item must
parse as a written number or quantity.** A list naming beliefs is
declined outright and falls through to the machinery that owns
beliefs, so "the sum of the tower height and the keep height" keeps
§11.42's voice while "the sum of 3, 5 and 10" gets this one. That is
§11.75's lesson applied *before* the bug rather than after it —
which is the only real use of a lesson.

Everything else is inherited rather than rebuilt. Units ride the
list the way they ride the operators: a same-family list converts
and reads in the largest unit present, a mixed-family list refuses
by name, and a bare number beside a united one refuses because
assuming its unit would be invention. Exactness is §11.76's — the
average of 1, 2 and 4 is 7/3, not 2.3333333333333335.

The gate (`experiments/list_gate.py`) measured nothing on its first
run (nothing varied) and was fixed §11.35's way. **Run two: 0.000 ->
0.804, +0.804 at 11.27x seed sd, zero wrong values, zero undefined
answers, zero boundary answers moved** — the 0.196 being the median
ceiling, a deliberate omission rather than an oversight (an
even-length median needs an ordering decision nobody has asked for,
and inventing one would be the convention §11.80 refused for "300 +
20%"). Criterion 4 is what the rung exists to protect, and it held:
belief lists, fact aggregates, and the single item that is not a
list answered identically in both arms, every time.

### 11.86 The question that needed no memory (failed, kept)

§11.85 registered an opportunity with a number. This section cashes
it, and the answer is no — which is worth more than a yes would have
been.

The diagnosis finished first. At 4,000 facts with the store flushed
as ordinary use leaves it, Recall was forming ~25 candidate keys for
"what is 12 + 3 * 4?", paging a bucket from disk for each, ~50 ms,
finding zero facts every time. The candidate ladder builds keys from
whatever words a question contains, and an expression contains none.
So Recall was taught to skip the ladder when the input is nothing
but numbers, operators and parentheses — the predicate being
§11.76's literal reader with no memory behind it, which is the same
claim said in code.

**It worked, and it bought nothing.** Parity was perfect — zero
response differences and zero recalled-fact differences across a
corpus of expressions, belief arithmetic, chains, polar questions,
comparisons, superlatives, aggregates, history, statements and chat
— and expression questions went from 21.20 ms to 1.14 ms, 18.5x. The
gate failed anyway, on its pre-registered criterion 4: belief
questions got *slower*. The decisive measurement was one the
criteria never asked for:

```
ladder   total corpus    1497.5 ms (sd 21.8)
skip     total corpus    1530.3 ms (sd 33.9)
saved per corpus: -32.8 ms (sd 50.4) over 24 questions
```

The skip makes expression questions 20-60x faster and the whole
session no faster at all, because **the twenty-five bucket loads it
removes were warming the bucket cache for every question that
followed.** The work was not wasted; it was prefetch wearing the
costume of waste.

That is the finding, and it outlives the optimisation: **the price
of a question at this density is the bucket cache, not the candidate
ladder.** A question that pages twenty-five buckets pays for them
and leaves them warm; a question that pages none pays nothing and
leaves the next question to pay. Any real win has to be at the
resident-set level — a larger or smarter cache, or an eviction
policy that knows what a session keeps asking about — and
per-question cleverness above it just moves the same milliseconds
around. Registered for whoever takes that on.

The mechanism is switched off and kept, in §11.60's tradition, with
the reason written beside the switch so it is not rediscovered as an
idea; the gate turns it on for its treatment arm, so the result
stays reproducible.

### 11.85 The arithmetic capstone

§11.76–§11.84 built arithmetic one rung at a time, and every one of
those gates ran at a small world, which is where mechanisms are
debuggable. §11.37's lesson governs what has to happen next: found
is not believed, and clean-room success means nothing until density
has its say. This is that run, in §11.74's capstone tradition — one
colliding world of 5,818 facts, every arithmetic form asked against
it, floors and controls and prices.

**PASS on the first run.** Ten forms at 1.000 (literal, percent,
power, exact root, enclosed root, quantity, reached operand,
comparison, polar arithmetic, rounding), **zero fabrications across
all eight refusal families**, zero false bounds. An operand nobody
holds, an operand that is a denial, an operand that holds no number,
a unit times a unit, a division by zero, a percentage of nothing, a
relation no vocabulary covers, and zero to the power zero were asked
eighty times between them and answered with a number zero times.

The floors are 1.000 deliberately. Arithmetic is deterministic — an
expression that evaluates correctly at twenty facts evaluates
correctly at ten thousand, because nothing about the store enters
the computation. What density CAN break is the part that touches the
store, so a miss here would be a mechanism bug rather than a density
effect, and the floors leave no room to hide one.

**The price line found something, which is what price lines are
for.** Literal arithmetic — two additions and a multiplication over
written numbers, touching no belief at all — cost 38.9 ms, which is
absurd on its face, so it was chased rather than reported:

```
calculate.evaluate alone :    0.016 ms
memory.find_facts alone  :    0.190 ms
whole pipeline           :   26.517 ms

Recall                 28.213 ms      (per stage, at 4,000 facts)
Learn                   3.063 ms
Reason                  0.180 ms
Route / Perceive / Respond  under 0.03 ms each
```

The arithmetic is a rounding error in its own cost; retrieval is a
rounding error too; and **the Recall stage is ~90% of the price of a
question it cannot possibly help with.** "what is 12 + 3 * 4?" names
no belief, and Recall runs before any branch decides whether the
question needs recall at all. Registered here with its number, in
the §11.45 tradition where the frontier clock earned its 3.2x only
after parity was proved — a measured opportunity, not a change.

### 11.84 Rounding is a request

§11.76 made exactness the rule, and by itself that is a little
unhelpful: sometimes two decimal places is exactly what a person
wants, and they are not confused about what they are asking for. So
rounding is a REQUEST — "to 2 decimal places", "to four decimal
places", "to the nearest whole number" are parsed off the tail of
the question, the arithmetic still runs exactly, and only the ANSWER
is rounded. Three properties keep this from undoing §11.76:

- The answer says it was rounded and names what from: "100 / 3 =
  33.33 — rounded to 2 decimal places; the exact value is 100/3."
- A value needing no rounding says that instead: "1 / 2 = 0.50 —
  exact at 2 decimal places, so nothing was rounded."
- Nothing is rounded unless it was asked for. Without a request,
  100/3 stays a fraction and sqrt 2 stays an enclosure.

The tie rule is a decision rather than an artefact: a half rounds
away from zero in exact rational arithmetic, so 0.125 is 0.13 and
-0.125 is -0.13. A rounded ROOT is proved rather than approximated —
the root is enclosed at increasing precision until both bounds round
to the same decimal, so the answer is right because two proved
bounds agree on it. And a request this rung cannot serve is refused
by name: "I can round to a number of decimal places or to the
nearest whole number; that is a different rule and I do not have
it," rather than reading "to" as an operand and refusing about the
wrong word.

The gate (`experiments/rounding_gate.py`) took three runs, and **the
mechanism was wrong once, the harness twice.** Run one: "to 2
decimal places" was answering "0.5" and "10", printing the shortest
exact form when somebody had named a precision — a request for two
decimal places is answered in two decimal places, because the
padding says how far the claim goes. The same run also exposed the
harness marking every division as needing rounding, when 300 / 6 is
exactly 50: **whether a case rounds is arithmetic, not a label the
gate gets to choose.** Run two came back clean and failed anyway on
zero variance, fixed §11.35's way. **Run three: 0.000 -> 0.802,
+0.802 at 14.51x seed sd, zero wrong roundings, zero unlabelled
approximations, zero unrequested answers moved** — the 0.198 being
the significant-figures ceiling, scored zero for both arms.

Criterion 3 is what keeps §11.76 intact. A rounded number that does
not say it is rounded is exactly the failure this subject was built
to refuse: **it is 1.4142135623730951 wearing a shorter coat.**

### 11.83 Asking whether the arithmetic holds

The system could compute "2 + 2" and could compare two beliefs, and
could do nothing with the question that joins them:

```
> is 2 + 2 = 5?
  I don't hold anything on that yet.
> is 3 * 4 greater than 10?
  I can't compare those: I hold nothing for '3 * 4'.
```

The second refusal is the honest shape of the gap: '3 * 4' really is
not a key, and saying so is true and useless, because the question
carries its own left-hand side and the reader that evaluates it was
already in the building. A comparison side may be an EXPRESSION now,
and the branch has learned equality — "=", "equals", "equal to" —
which pivots exactly as "taller than" does, so both sides go through
the same machinery: units, conversions, derived operands and all.
The equals sign is split out of whatever it was written against, so
"is 2+2=4?" reads.

Everything composes because nothing was special-cased:

```
> is the tower height * 2 greater than 500 meters?
  Yes - tower height * 2 is 600 meters, 500 meters (confidence 0.60)
```

That multiplies a belief and keeps its unit (§11.78), compares
against a quantity nobody stored (§11.82), and reads the whole left
side as an expression (here) — three rungs built separately, for
separate reasons, meeting in one sentence because each was built as
a rule rather than a case.

One bug was fixed on the way in, and it was §11.78's own: its reader
was taking polar questions, so "is 2 + 2 equal to 4?" refused with
"I hold nothing for 'is'" — a refusal about the wrong word entirely.
A polar lead asks whether something HOLDS, which is this rung's
question, and the reader declines it.

The gate (`experiments/polararith_gate.py`) ran `_EXPRESSION_OPERANDS`
off against on. **PASS on the first run: 0.341 -> 1.000, +0.659 at
8.52x seed sd, zero wrong verdicts, zero unreadable sides answered,
zero comparisons moved, zero plain polar answers moved.** The
equality voice is deliberate: "Yes - 4, 4" is a machine agreeing
with itself, while "Yes - 2 + 2 is 4" and "No - 2 + 2 is 4, not 5"
are the shapes the polar family has used since §11.48, so the newest
question sounds like the oldest one.

### 11.82 A coin that always said No

Found by asking the shipped system a question in the words people
actually use, with `tower height 300 meters` and `keep height 100
meters` in the store:

```
> is the tower height greater than 200 meters?
  No - tower height is 300 meters, not greater than 200 meters
> is the tower height greater than 500 meters?
  No - tower height is 300 meters, not greater than 500 meters
> is the tower height less than 500 meters?
  No - tower height is 300 meters, not less than 500 meters
```

Every one No, in every direction, and three of those six probes were
wrong. §11.54 owns comparatives and had ended exactly this failure —
"taller than" answered correctly throughout the same probe — but
"greater", "more" and "less" were not in its vocabulary, so the
questions fell past it to the polar branch, which compares the CLAIM
to the stored VALUE as strings. "300 meters" is never the string
"greater than 200 meters", so the answer was always No. **§11.54's
coin had not been removed; it had been narrowed to the words nobody
had thought of yet.**

Two fixes, and the second is the one that lasts. The missing words
join the vocabulary (greater, more, less, fewer). And **a comparison
the comparative branch declined is REFUSED, not answered**: if
"than" is in the claim and §11.54 did not take the question, the
relation is one this system cannot compare by, and it says so — "I
can't tell: I don't know how to compare by 'wider'." Adding four
words fixes four words; that line fixes the ones nobody has thought
of yet. A third thing came free: the right-hand side may be a
WRITTEN quantity rather than a key, since §11.78's reader already
knows how to read "200 meters" and convert it.

The gate (`experiments/comparison_gate.py`) runs against the system
exactly as it shipped — guard off, words removed. **PASS on the
first run: 0.383 -> 1.000, +0.617 at 3.23x seed sd, zero wrong
verdicts, zero unknown relations answered, zero old-vocabulary
answers moved, zero plain polar answers moved.**

The baseline's 0.383 is not a score, it is the shape of the bug: a
branch that always says No is right exactly as often as the truth
happens to be No, so **the number measures the worlds, not the
system**. That is what makes this failure mode dangerous rather than
merely wrong — it looks like a system with opinions, it is right in
nearly two cases out of five, and nothing about a single answer
distinguishes a computed No from a No that was never computed.

### 11.81 A root that has no value still has a place

Powers and roots are where "say exactly what is true" stops being a
style and starts costing something. A whole power is easy and exact:
2^10 is 1024, 2^-3 is one eighth, the sign sits outside the power so
-2^2 is -4, and the exponent binds tighter than multiplication
because that is what everyone writing "3 * 2^4" means. A root is not
easy. **The square root of two has no exact value** — not as a
decimal, not as a fraction — and the standard option answers it
anyway, with 1.4142135623730951, whose square is 2.0000000000000004.

So this rung refuses to name it and says where it is instead:

```
> what is the square root of 2?
  sqrt 2 is between 1.414213562 and 1.414213563 - that value is
  irrational, so those are proved bounds, not the number.
```

The bounds are *proved*: computed by integer arithmetic
(`math.isqrt` and an integer nth root, never a float) and then
verified as rationals — the low bound raised to the degree is
checked to be at most the radicand, the next step up checked to
exceed it. A root that IS rational is given exactly (sqrt 16 is 4,
sqrt 0.25 is 0.5, cbrt -8 is -2), so the enclosure is reserved for
numbers that need it. Three shapes refuse with their reasons: a
negative radicand under an even root, zero to the power zero ("a
convention, not a value — different fields choose differently, and I
will not choose for you"), and an exponent past a thousand. Units
refuse by the rule §11.78 already held — raising meters to a power
would name a unit no definition covers.

The gate (`experiments/power_gate.py`) measures against the standard
option honestly built (same grammar, same refusals, `**` and float
roots). **PASS on the first run: 0.702 -> 1.000, +0.298 at 3.56x
seed sd, zero wrong values, zero irrationals named, zero false
bounds, zero refusals missed.** The false-bounds line is what keeps
this from being a worse answer dressed as a better one: the gate
checked every enclosure itself in exact rational arithmetic, because
a wrong bracket would be a firmer lie than the float it replaces.
The enclosure is shorter than 1.4142135623730951, and it is true.

### 11.80 A hundredth part, taken

"what is 20% of 300?" is basic arithmetic by any account, and the
system could not read it — the per-cent sign did not tokenize, so
the question fell through the whole ladder. It reads now, and reads
*structurally*: a percentage is a number over a hundred, and "of" is
the multiplication it is waiting for, so nothing downstream needs a
special case. That is why percentages compose with everything the
previous rungs built — "15% of the tower height" is 45 meters
through §11.78's units, and "20% of the west tower height" reaches
through §11.79's chain and says so.

**One shape refuses, and it is the interesting one.** "300 + 20%"
has no answer. Twenty per cent OF WHAT is a question the sentence
does not answer, and the near-universal convention — take the
percentage of the left operand, so 360 — is a convention rather than
an arithmetic. Every calculator that answers 360 is guessing at the
speaker's meaning, which is exactly what the branch built to be
exact must not do.

The gate (`experiments/percent_gate.py`) took two runs. **The first
measured nothing**: the baseline could not read a single percentage
and the treatment read them all, so every seed returned the same
contrast and the seed sd was zero — a FAIL under the house rule,
which is not softened when it is inconvenient. The fix is the one
§11.35 prescribes: score in the cases nobody can win, here the
inverse question ("what percent of 300 is 60?", which neither arm
reads — the reader takes expressions, and that is a sentence).
**Run two: 0.000 -> 0.832, +0.832 at 17.28x seed sd, zero wrong
values, zero conventions taken, zero voice splits against the
spelled-out arithmetic, zero ordinary questions moved** — the 0.168
being the inverse question, scored zero for both arms.

Criterion 4 says the most about the design: every "N% of X" was also
asked as "X * N / 100", and the two agreed every time. That is what
it means for a percentage to be structural rather than a special
case — the reader turns it into arithmetic the grammar already had,
so there is no second code path to disagree with the first.

### 11.79 An operand the store can reach

§11.78 closed with a ceiling on the record, and it was really a
disagreement. One store answered two questions about the same fact
two different ways:

```
> is the west tower taller than the south tower?
  Yes - west tower height is 900 meters (derived via spirekind),
  south tower height is 450 meters
> what is the west tower height * 2?
  I can't compute that: I hold nothing for 'west tower height'
```

Both sentences are about `west tower height`. §11.55 reaches it
through `west tower kind is spirekind`; the arithmetic said it did
not exist. That is §11.75's failure again — one matrix, two voices —
so the arithmetic operand now goes through the same chain machinery
under the same three guards (no modifier-rescued subject, no derived
denial, no empty conclusion), is marked in the same words ("derived
via spirekind"), and carries its diluted confidence into the
minimum: a number reached through two facts is not as good as a
number that was stated, and the answer says so.

The gate (`experiments/derivedoperand_gate.py`) ran `_DERIVE_OPERANDS`
off against on. **PASS on the first run: 0.662 -> 1.000, +0.338 at
3.64x seed sd, zero wrong values, zero derived denials given a
number, zero voice splits against the chain machinery, zero held
operands moved.** The fourth line is the one the rung exists for:
every derived operand was asked twice — once inside the arithmetic,
once on its own — and required to be the same number. The ceiling is
closed in the direction that removes a disagreement rather than
adding a capability.

### 11.78 Arithmetic over what is believed

§11.76 could evaluate what the speaker wrote; it could not use what
the system knows. "what is the tower height times 3?" answered
"tower height is 300 meters" — a recall of one operand offered as
the answer to an expression, which is §11.29's coverage failure
wearing an operator: the question's tokens were covered, the
question was not.

An operand is now any of three things — a written number, a written
quantity ("300 meters"), or a held belief — and units ride the
operators. Addition needs one unit, reached through §11.42's
definitions, and reads in the LARGER of the two (the rule the
combine path already answers by); multiplication takes at most one,
because meters times meters would name a unit no definition covers;
division of two quantities in one family gives a plain ratio, which
is the useful and honest answer. Conversions apply the definitions
EXACTLY: §11.42 stores its factors as floats, and a float is the
wrong shape for a definition — 0.3048 is what a foot is in meters,
the double nearest 0.3048 is not — so they are read back through
their decimal spelling. A result resting on a belief is a
derivation, marked inferred, naming its premises, carrying the least
confident of them.

**Placement is the safety argument.** §11.76's literal reader is
provably safe anywhere, because an expression of nothing but numbers
cannot be another branch's question. This one reads WORDS, and words
are what every other branch is made of — so it runs after every
word-reading question form and before recall: late enough to steal
nothing, early enough that an expression is never answered by one of
its own operands. An expression where nothing resolves is not
arithmetic at all: "what is the plus size?" reads "plus" as an
operator and must fall through untouched, so a refusal is spoken
only when something else in the expression did resolve.

The gate (`experiments/quantity_gate.py`) took two runs, and the
first was the harness's fault: it drew "cross-family" refusal cases
by picking a different ATTRIBUTE, and height and length are both
measured in meters, so two same-family sums computed correctly and
were scored as inventions. An attribute is not a family. **Run two:
0.000 -> 0.800, +0.800 at 18.46x seed sd, zero wrong values, zero
undefined answers, zero voice splits against the combine path, zero
ordinary questions moved.** The 0.200 is the derive ceiling, scored
in rather than excused — §11.55 derives a comparative operand the
store does not hold, this rung does not, so "the west tower height *
2" refuses by naming the key it looked for, and both arms score zero
there.

**One matrix, one voice** is now a measured line rather than a
lesson: every two-belief sum was asked twice, once through this rung
("X + Y") and once through §11.42's combine path ("the sum of X and
Y"), and required to agree on the number and the unit. §11.75 was
caught because two branches answered one question differently; this
makes that class of bug measurable instead of noticeable.

### 11.77 The unit that fell off

Found while building §11.76, by asking the shipped machinery a
question it had always been able to answer:

```
> what is the sum of the tower height and the keep height?
  keep height + tower height = 500 - inferred, not stored:
  keep height is 200 meters; tower height is 300 meters
```

500 what. Both premises say meters and the answer says a bare
number. The cause is one line of §11.42's combine path: the unit
suffix was attached to the CONVERSION rather than to the result, so
"300 meters + 1 kilometer" answered "1.3 kilometers" correctly while
"300 meters + 200 meters" fell through the same branch with an empty
suffix. All three combine shapes inherited it — sums, differences,
and the larger/smaller verdict ("tower height (300) is the larger of
the two") — while the aggregate path (§11.57) had it right all
along, which is what made the split visible from outside: one store,
two voices, for the same arithmetic. The result unit is a property
of the result now; the conversion note stays exactly where it was;
unitless operands stay unitless, because inventing a unit for
"300 + 200" is the opposite failure.

The gate (`experiments/unitkeep_gate.py`) ran `_SUM_KEEPS_UNIT` off
against on. **PASS on the first run: 0.528 -> 1.000, +0.472 at 1.87x
seed sd, zero numeric values moved, zero units invented, zero
converted-or-refused answers changed.** The never-moves line is the
one that matters: every number in every answer was identical across
arms — this changes what the answer says, not what it computes.

**Why a green suite shipped it.** The same-unit case *was* pinned —
`test_same_unit_answers_carry_no_conversion_label` — but the pin
asserted only what the answer must NOT say, because that was the
claim being made the day it was written, and a test checking for the
absence of a label cannot notice the absence of a unit. Nothing else
in 1,697 tests looked at that sentence: **the fix broke no pin,
which is the tell.** A green suite proves the claims someone thought
to write down, and "the answer says what it is an answer in" was
never one of them until the question was asked in plain English and
the answer read as a reader rather than as its author. The pin has
been completed rather than replaced.

### 11.76 Arithmetic that is exactly right

The system could combine two STORED numbers since §11.42 but could
not add two: "what is 2 + 2?" fell through the entire question ladder
to "I don't hold anything on that yet" — honest, and useless.
`reason/calculate.py` reads an arithmetic expression and evaluates
it, with precedence and parentheses structural rather than
special-cased (`expr := term (('+'|'-') term)*` and down), word
operators mapped to the same operators, and signs handled by the
grammar. A bare expression is admitted as a question in Perceive:
"6 * 7" asks in a shape no interrogative lead covers, and an
expression is self-identifying — nothing but numbers, operators and
parentheses — so admitting it steals nothing, because a statement
carries words.

**It evaluates over exact rationals, not floats**, and that is the
whole design choice. The standard option cannot hold a tenth: it
answers "what is 0.1 + 0.2?" with 0.30000000000000004, a number that
is not the answer. A system whose entire discipline is "say exactly
what is true" cannot answer a question about tenths with a number
that is not the answer. Decimals parse exactly, the renderer prints
an exact decimal when the denominator allows one and an exact
fraction when it does not ("1/3", never 0.333…, which is a different
number), and division by zero refuses aloud.

The gate (`experiments/arithmetic_gate.py`) does not measure against
an absence. **Its baseline arm is the standard implementation,
honestly built** — same reader, same grammar, same refusals,
evaluating in binary floating point — and ground truth comes from
the gate's own expression TREES, walked with `Fraction`, never from
the parser under test. **PASS on the first run: 0.443 -> 1.000,
+0.557 at 3.69x seed sd, zero wrong answers, zero undefined
divisions answered — while the float arm printed a number that was
not the answer 45 times and answered 8 undefined divisions.** The
two failures differ in kind: `0.1 + 0.2 = 0.30000000000000004` is a
wrong digit, but `7 / (0.1 + 0.2 - 0.3) = 126100789566373888` is a
wrong BRANCH — the divisor is zero, the float arm reaches a residue
near 1e-17 instead, and reports a hundred and twenty-six quadrillion
for an expression that has no value at all. Ordinary questions
answered identically in both arms and in a control session with the
branch switched off entirely.

### 11.75 The comparative word carries its attribute

"is the west tower taller than the south tower?" refused with "I
hold nothing for 'south tower'" — while the store held "south tower
height is 450 meters" — and "which is the tallest?" answered nothing
over three stored heights. The word "taller" NAMES the attribute and
nobody read it: both branches required "height" spelled out, a shape
nobody speaks. Now taller/tallest say height, heavier/heaviest and
lighter/lightest say weight, longer/longest say length — and only
those: the polysemous words (bigger, larger, smaller, highest,
lowest) name no single family ("highest price"), so implication
there would fabricate specificity, and they keep requiring the named
attribute. The same test excluded one of our own candidates before
any run: "shorter" spans height and length ("the shorter rope") and
was dropped from the map. Exact bare keys still win first; the
appended key is tried only when the bare one misses; §11.55's derive
runs on the appended key, where the attribute's chain actually
lives; the refusal names the key that was tried.

The gate (`experiments/impliedattr_gate.py`) ran `_IMPLIED_ATTRS`
off against on. **PASS on the first run: 0.000 -> 0.700, +0.700 at
9.48x seed sd** — the baseline answered NO natural shape at all, the
treatment answered every winnable one (0.700 is exactly the winnable
share; unwinnables score zero for both arms by §11.35's arithmetic)
— **zero ambiguous words answered, zero unwinnables answered, zero
named-form mismatches, zero bare pairs shadowed.** The derive trail
rode through: "Yes — west tower height is 900 meters (derived via
spirekind), south tower height is 450 meters" — implication, chain,
and comparison composing in one verdict.

### 11.74 Two statements joined are two statements

"the tower material is iron and the bridge material is steel" hit the
first " is " split and stored the ENTIRE second clause inside the
first value — 'tower material' = "iron and the bridge material is
steel" — corruption narrated as success, the second fact drowned
where no question would find it. A clause splits only where BOTH
sides independently parse as statements, so the fix cannot overreach:
"the motto is live and learn" never splits (its right side carries no
verb), and each clause runs the complete statement machinery through
§11.71's plumbing — polarity per clause, revision notices per clause.

The gate (`experiments/clause_gate.py`) ran `_CLAUSE_CONJUNCTIONS`
off against on. **PASS on the first run: 0.361 -> 1.000, +0.639 at
4.59x seed sd, zero statements swallowed, zero and-values split —
while the baseline swallowed 20**, the shipped corruption measured
beside the fix. The statement grammar is complete in both directions:
subjects conjoin (§11.71), relations distribute (§11.72), questions
mirror (§11.73), clauses split (here) — four forms, one machinery,
and no shape of "and" left that silently loses or corrupts a fact.

### 11.73 One grammar for teaching and asking

§11.72's question-side mirror: "what is the tower and the bridge
material?" hedged with a junk curiosity while its full-subject form
answered both parts. The rule applies verbatim to `_compound_parts` —
the final part's trailing relation completes every single-token
question part, the same two lines hold (no relation anywhere refuses;
multi-word entities stay the ambiguity ceiling) — and teaching and
asking finally share one grammar: a sentence that stores two facts
can be asked back in the same shape.

Building it caught a pre-existing §11.48 gap: the compound formatter
predates polarity and asserted a denied value positively ("dome
material is steel" for a stored "not steel") the moment the ellipsis
routed a denial through it. The formatter speaks polarity now. The
gate (`experiments/qellipsis_gate.py`) ran `_QUESTION_ELLIPSIS` off
against on: **PASS on the first run, 0.278 -> 1.000, +0.722 at 4.04x
seed sd, zero denials rendered bare, full-subject compounds and the
arithmetic combine identical at 1.000 in both arms.**

### 11.72 The relation stated once distributes

§11.71's ceiling, cashed for exactly the shape it is honest in: "the
tower and the bridge material are iron" names the relation once, in
the final part, and standard ellipsis completes every SINGLE-TOKEN
conjunct with it — nothing guessed, nothing dropped, negation riding
the distribution. Two lines stay unmoved: a conjunction where nobody
names a relation still refuses (there the distribution really would
be invention), and multi-word entities stay the sharpened ceiling —
token count cannot say whether "keep" in "the north tower and the
south keep material" is name or relation, so §11.47's spirit refuses
the ambiguity while the full-subject path stores multi-word parts
exactly as their singular equivalents would.

The gate (`experiments/ellipsis_gate.py`) took three runs: its first
worlds used two-word entities — measuring a claim the mechanism never
made, and sharpening the ceiling in the process — and its second run
hit the third slicing-overflow in the book (fixed slots end the
class). Then: **PASS, 0.333 -> 1.000, +0.667 at 3.83x seed sd, zero
relation-free conjunctions stored, full-subject conjunctions
identical in both arms.** A fourth ceiling registered and cashed
inside one session; the adjacency notes also learned to speak once
per base per turn along the way.

### 11.71 Conjunctive statements teach all their facts

"the tower material and the bridge material are iron" offered two
facts and taught zero, silently — " are " never parsed, the intent
classifier read the sentence as chat, and the teacher had no way to
know the lesson was dropped. The plural form is admitted exactly as
wide as it is honest: " are " counts as statement-shaped only beside
" and " (a conjunctive sentence is plural by construction; a bare
" are " sentence is conversation), full-subject parts split and each
takes the shared value through the COMPLETE statement machinery — a
shared `_learn_statement` helper, so polarity distributes ("... are
not steel" denies each part), a revising part narrates its §11.53
notice inside the conjunction, and an adjacent part its §11.63 note.
The elided form ("the tower and the bridge are iron") stays the
registered ceiling: distributing an attribute the parts do not name
would be guessing, and the refusal says so. A key containing " and "
is never stored, ever.

The gate (`experiments/conjunction_gate.py`) ran
`_CONJUNCTIVE_STATEMENTS` off against on. **PASS on the first run:
0.000 -> 0.836 at 6.45x seed sd, zero junk keys, singular statements
identical in both arms.** A teacher can finally say two things in one
sentence and have both of them land, each through the full machinery
it would have met alone.

### 11.70 Declining a derivation names its premises: the family whole

The testimony family's last door: "no" after a DERIVED answer.
§11.69's protocol does not transfer — nothing was stored, so nothing
lowers, and a bare "no" cannot say WHICH premise it blames, so no
premise confidence moves either: blaming an unnamed premise would
invent the accusation. What the refusal earns is the trail, §11.31's
thesis serving correction: "Noted - not consolidating that. The
derivation rested on: tower material is iron; iron hardness is 490
units. If one of those is wrong, restate it" — the wrong premise
restated closes into §11.53's narrated revision, the chain re-derives
from what is NOW believed, and "yes" consolidates the corrected
conclusion. Contest, correct, re-derive, confirm — the full cycle,
every step spoken.

The gate (`experiments/decline_gate.py`) ran `_DECLINE_DERIVATIONS`
off against on. **PASS on the first run: 0.472 -> 1.000, +0.528 at
3.07x seed sd, zero unnamed premises blamed, zero declined
conclusions stored, consolidation untouched in both arms.** The
testimony family is whole: yes consolidates a derivation or confirms
an assertion, no declines a derivation naming its premises or
contests an assertion dropping it to doubt — four responses to two
words, chosen by what was actually pending, and a bare word with
nothing pending still moves nothing anywhere.

### 11.69 No is testimony too: doubt spoken, never absence

§11.68's mirror was still mush — a polar "Yes - tower material is
iron" answered with the user's "no" fell to the chat fallback, and
testimony AGAINST a belief vanished. The mirror closes with an honest
asymmetry: a contest drops the belief to doubt (0.30) but does NOT
delete it, because a bare "no" names no replacement and deleting on
it would invent an absence. The drop is said aloud, the correction is
asked for ("tell me what it is"), and a following statement closes
into §11.53's narrated revision — doubt, ask, revise, each step
spoken. The testimony pair is complete: yes raises to 0.90, no lowers
to 0.30, both aloud, neither deletes, and a bare word with nothing
pending moves nothing.

The gate (`experiments/disconfirm_gate.py`) ran
`_DISCONFIRM_TESTIMONY` off against on. **PASS on the first run:
0.278 -> 1.000, +0.722 at 3.33x seed sd, zero deletions, zero moved
strays, the §11.68 confirmation floor untouched in both arms** —
every contest at exactly 0.30, every loop closing into the narrated
revision. The user's agreement and disagreement finally weigh exactly
what the epistemics always said they should.

### 11.68 Yes reaches the stored fact: testimony at the surface

`confirm_fact` has argued since it was written that "yes, that is
correct" is stronger evidence than hearing a value again in passing —
reinforcement nudges by 0.1, testimony asserts outright — but nothing
wired the mechanism to the surface: a polar "Yes - tower material is
iron" followed by the user's "yes" answered "I have nothing on that
yet". The loop closes: an assertion of a held belief (polar yes,
choice pick) leaves a one-turn pending confirmation — §11.33's
freshness discipline, §11.16's trap avoided — and "yes" raises the
fact to 0.90 as direct testimony, said aloud. The two strengths never
conflate and they COMPOSE: a confirmation followed by a later passing
mention reads 1.00, each contribution at its own strength. The
affirmation word now reaches three pending kinds: a derivation
consolidates, an assertion confirms, and nothing pending stays
nothing.

The gate (`experiments/testimony_gate.py`) ran `_CONFIRM_TESTIMONY`
off against on. **PASS on the first run: 0.442 -> 1.000, +0.558 at
3.99x seed sd, zero strengths conflated, zero strays moved, the
§11.33 consolidation floor untouched in both arms** — every
confirmation at exactly 0.90, every passing restatement at exactly
0.70, the docstring's eleven-section-old claim finally measured where
users live.

### 11.67 Compound disjuncts compare whole

§11.66's registered ceiling, cashed within the session that registered
it — the third such: the branch refused multi-word disjuncts rather
than half-parse a compound alternative, but fold-equality compares a
compound WHOLE or not at all, so there was never anything to
half-parse. "is the spire architect wren or wren the younger?" picks
the held compound by name; "reinforced steel" against a held "steel"
answers Neither, because a qualified value is a different value —
§11.47's lesson in question form; the bare base does not match the
compound it abbreviates; and absence never picks a side, however many
words the side has. The first probe caught the spoken lead using the
article-stripped PARSE ("Wren younger") — the lead now names the
belief as held ("Wren the younger").

The gate (`experiments/multichoice_gate.py`) ran `_CHOICE_MULTIWORD`
off against on. **PASS on the first run: 0.175 -> 1.000, +0.825 at
7.66x seed sd, zero sides elected without grounds**, §11.66's
single-word forms identical at 1.000 in both arms.

### 11.66 Choices: the held disjunct answers by name

"is the tower material steel or iron?" reached the polar matrix as
one compound claim, matched nothing, and answered No — to a question
whose true answer was on the table. Disjunctions have an owner now:
the offered alternatives are VALUES, and the held one answers by name
("Iron - tower material is iron"), "Neither" answers with the actual,
a stored denial rules out its own value without electing another
("Not steel, at least" — ruling out is not picking, because a denial
of one value affirms nothing about another), and an unheld subject
falls through to the hedge. The zero-invention line one more way:
absence never picks a side. Multi-word disjuncts are the registered
ceiling — the branch refuses rather than half-parse a compound
alternative.

The gate (`experiments/choice_gate.py`) ran `_POLAR_CHOICES` off
against on. **PASS on the first run: 0.235 -> 0.901, +0.667 at 5.92x
seed sd, zero sides elected without grounds**, plain polar identical
at 1.000 in both arms — and the baseline's pick-kind zero is the
shipped wrong No, measured beside the fix. The polar family covers
assertion, denial, derivation, comparison, and choice: one matrix,
and absence never says yes, no, or picks a side through any door of
it.

### 11.65 What happened: the retraction, askable — the tense family closes

§11.64 opened the revision record's read side; this unit opens the
other ending. A consolidated fact that truth maintenance takes down
(§11.30: a conclusion does not outlive its premise) simply vanished —
the user who watched it consolidate found a hedge where it stood,
though the retraction episode records exactly what happened and why.
The data had the same gap §11.64's polarity fix closed: retraction
episodes carried no tags and could not be found by key. Tagged now, a
past-tense question over a subject no longer held answers from the
record — "I no longer hold tower hardness: retracted — premise 'tower
material' was revised. It had been consolidated from a derivation,
and a conclusion does not outlive its premise" — while the PRESENT
tense honestly re-derives forward from the new premise instead of
mourning the old conclusion. The zero-invention line extends
unchanged: only a recorded ending answers; an ending invented is a
history invented.

The gate (`experiments/retraction_gate.py`) ran the flag's two arms
over consolidate-then-revise sequences built live through the affirm
flow. **PASS on the first run: 0.369 -> 1.000, +0.631 at 7.82x seed
sd, zero endings invented**, with the §11.64 histories and the
present untouched in both arms. The tense family is complete: what is
(recall and derivation), what was (§11.64), what happened (here), and
why (§11.51) — one record, four doors, nothing invented through any
of them.

### 11.64 What was: belief history, askable — and never invented

Every change of mind has been an episode since §11.16's era, a spoken
notice since §11.53, and a queryable provenance since §11.51 — but
"what was the tower material?" answered with the present: the record
was write-only. Past-tense questions ("what was X?", "what did X used
to be?") now answer from the revision episodes — every past belief
named in order with its count ("It was iron, then steel; it is bronze
now, 2 revision(s) on record"), polarity spoken where the past was a
denial ("It was not steel") — which required a data-shape fix found
in passing: the revision episode logged bare values, so a §11.48
flip's history would have read "was steel" when the belief was "not
steel". Episodes now log spoken forms; old episodes keep their shape.
The line is §11.51's one tense back, at zero tolerance: **history is
recorded, never invented** — a fact stated once answers "I have no
record of it ever being different", and an unheld subject gets a
hedge, never a past.

The gate (`experiments/history_gate.py`) took two runs: the first
failed its own present floor at 0.431 in both arms identically — the
builder SHUFFLED its statements, and in a history world the statement
sequence is the semantics; the harness lesson joins §11.58's article
keys and §11.63's slot collision (a world must be built the way the
tested thing actually happens). Order preserved: **PASS, 0.127 ->
1.000 at 11.16x seed sd, zero histories invented, the present
untouched at 1.000 in both arms.** The belief store answers in three
tenses now — what is, what was, and why — all from one record, none
of it invented.

### 11.63 The neighbor named: adjacency spoken, never merged

§11.53 measured revision detection's exact-key ceiling and scored it
into its own gate: "the old tower material is steel" against a held
"tower material" stores a NEW fact and narrates nothing — a more
specific subject is a different belief. But silence about the held
base invites the reader to assume the two are one. A new fact whose
leading-stripped base key is held now answers with the neighbor
spoken — "Noted: old tower material is steel. I separately hold:
tower material is iron" — polarity included when the base is a denial,
never as a revision, never a merge: different keys are different
beliefs, and now both are on the table. The three silences hold at
zero tolerance: fresh no-base facts, reinforcements, and revisions
(which keep their §11.53 notice and gain nothing else) carry no note,
because a note that fires on everything informs about nothing.

The gate (`experiments/nearkey_gate.py`) took three runs: the
variance rule (ninth "nothing varied"; the single-strip ceiling
scored in — a base two strips away stays unnamed, multi-qualifier
adjacency registered), then its own semantics floor caught a
world-builder slot collision (the near loop and the far kind shared a
name, and a setup line silently revised a recorded base — the
revision gate's entity-reuse bug, caught by the same kind of floor),
then **PASS: 0.000 -> 0.912 at 8.37x seed sd, zero false notes,
semantics 1.000 in both arms**.

### 11.62 The third adjective: the cap is policy, the floor is safety

The modifier family's registered rung, cashed the family way: §11.36
shipped tolerance at one dropped token, §11.46 moved the cap to two
behind the residual floor, and this unit moves it to three — the floor
untouched through all three moves. At most three library-unknown
tokens may be read as decoration; at least two known informative
tokens must survive, and convergence still demands a bridge. "The
ancient weathered crumbling tower melting point" reads three
adjectives away, names all three in question order, and converges
through the bridge; a fourth unknown refuses above, and a thin
residual refuses below, exactly as before.

The gate (`experiments/triplemodifier_gate.py`) ran cap-2 against
cap-3 with ~25% quadruple-modifier worlds as the unwinnable share and
both §11.46 decoy forms. **PASS on the first run: 0.000 -> 0.583 at
1.13x seed sd — narrowly, in §11.13's tradition, on the same dealt
hand §11.46 drew** (the identical seed positions draw the identical
5/12 quadruple share) — zero decoys of either form asserted, every
answer naming all three words. Three cap moves, one unmoved floor:
the cap is policy, the floor is safety, and the distinction has been
measured every time it mattered. The quadruple rung stays registered,
not promised.

### 11.61 The fifth fact: the second limit moved by measurement

§11.52 closed with the fifth fact "a measurement away from moving,
never a promise", and the open question was real in both directions:
either the structural rules carry a FOURTH bridge, or that is the
depth where §11.31's warning finally binds — and a FAIL would have
been the binding measured, recorded exactly the way a pass would be.
The depth-five gate (`experiments/depth5_gate.py`) asked it with the
candidate constants (`_MAX_HOPS = 4`, `_FLOOR = 0.05`) monkeypatched
per arm: entity -> agent -> place -> region -> realm -> attribute
worlds (five facts, four bridges), ~25% broken middles as the
unwinnable share, the full decoy family at the new depth, and four-,
three-, and two-fact chains plus recall as regression floors.

**PASS on the first run: five-fact chains 0.000 -> 0.896 at 6.96x
seed sd, zero decoy assertions at four bridges, every shallower floor
identical in both arms.** The constants shipped; the sixth fact
arrives at 0.03125 and stays structurally refused, a measurement away
like the two limits before it. The ladder protocol moved up with the
capability a second time — depth-5 converges without a rung, depth-6
closes with exactly one confirmation, and a bottom revision still
takes the rung down. Two limits moved by measurement in one session:
the § of record is not the constant but the method — pre-register the
risk, arm the candidate, and let the decoys decide.

### 11.60 The drowned bridge that wasn't: a negative result, recorded

After §11.59, the next clock-and-completeness candidate looked
obvious: bridge retrieval rides `find_facts` top-k ranking, a
single-token probe ("bronzite") has no bigram to address with, and a
terminal-slot target ("bronzite base", whose relation word never
appears in the question) should therefore DROWN once its token ties
across enough buckets — a silent recall ceiling. A prefix index
(first informative token -> the two-token prefixes starting with it,
persistent, add-only, §11.32-style lazy staleness) was built to make
the candidate set complete by construction.

It was shelved unadopted, because the failure it fixes could not be
demonstrated. Three increasingly hostile constructions — 20 drowners,
70 drowners across varied seeds, 70 drowners with per-seed bridge
values on a flushed store — all recovered the chain in BOTH arms, and
the instrumented trace showed why the hypothesis was wrong twice
over: bucket ties sort deterministically by shard id (not by luck),
which put the target fifth of 58 candidates, and inside consulted
buckets §11.44's reinforcement tie-break plus alphabetical key order
rank the short genuine key above every drowner. The §11.37 addressed
pair-probes, the deterministic tie, and the key ranking jointly cover
the case the index was designed for. (The trace also surfaced one
regime honesty note: before a store's first flush, a single-token
probe with no ranked candidates falls back to scanning every staged
bucket — retrieval is COMPLETE pre-flush and ranked post-flush, and
the post-flush regime is the one all three constructions failed to
break.)

Found is not believed applies to problems exactly as it applies to
solutions: a mechanism may only enter behind a gate whose baseline
arm actually fails, and a gain of zero over a healthy baseline is not
a unit — it is a shelf entry. The index implementation is recorded
here, not in the tree, so the next hypothesis about bridge drowning
starts from this measurement instead of re-deriving the hunch.

### 11.59 The depth clock: licensed by measurement, held to parity

"Always Complex it before optimization" is a standing rule, and
§11.58 finally licensed this unit by measuring the price: depth-4
chains at 726.5 ms/query, the largest number on the capstone's table,
with the profile putting 92% of it in index-search volume — every
activated node re-relaying in every synchronous round, recomputing
identical bridge probes, and shared bridge values re-querying the same
buckets (~1,627 `find_facts` calls per question). Two structural
call-eliminations closed it, both provably identical in output:
**frontier relaying** (after round one, only nodes that gained or
strengthened an origin last round relay — direct sources never change
after seeding, and equal-strength re-arrivals were already discarded
unread by the strict comparison) and a **per-spread probe memo**
(bridge probes depend only on the value and the fixed question).
`_FRONTIER_SPREAD` is the gate's arm switch.

The gate (`experiments/clock_gate.py`) runs §11.45's shape one tier
up: both arms against the SAME session on identical store bytes,
parity absolute and first — chains at every depth, modifier
tolerance, negated terminals, every decoy family, byte-identical
responses required — and the clock second. **PASS: zero divergences
on the whole battery, 270.3 -> 84.8 ms per depth-4 query (+185.5 ms
at 20.27x repetition sd)** — and the capstone, re-run under the
frontier, reads every floor still at 1.000, zero fabrications, and
**depth-4 at 347.4 ms** at 9,631 facts, down from 726.5. Rule 1 held
the whole way: a fast wrong answer is not an optimization, and a fast
differently-worded answer is not parity.

### 11.58 The density revisit: the session's arc at scale

§11.37's lesson governs its descendants: every capability of the
§11.45-§11.57 arc was gated at small worlds, where mechanisms are
debuggable — and clean-room success means nothing until density has
its say. The revisit gate (`experiments/revisit_gate.py`) is the
capstone the lesson demands: one colliding world of ~10,000 facts,
every question form of the session asked against it, fabrication at
absolute zero across every decoy family the session built, floors
pre-registered per form, and the full-scan price finally seen at
scale.

**Its first run indicted the harness.** The name pool ran dry at
2,731 facts while claiming ten thousand, and polar, why, and compare
scored 0.000 with second-long latencies while depth-4 sat at 1.000 —
the asymmetry was the diagnosis: the builder ingested keys WITH
articles through direct `remember_fact`, but the live statement path
strips them and the recall ladders look up stripped keys, so every
recall-backed form missed facts the token-based spread could still
see. A density harness must key its world the way live usage would.
Pools enlarged, keys stripped, and at 9,631 facts: **every floor at
1.000, zero fabrications** — depth-4 chains 726.5 ms (three
synchronous frontier rounds over a dense store: the expensive
capability, expensively honest), polar and why at 1.3 ms on the
addressed index, compare at 4.6 ms, and the full-scan forms saying
their price aloud (superlative 49.5 ms, aggregate 18.3 ms). The
session's arc survives density whole.

### 11.57 Aggregates: the family counted, totalled, and averaged

The enumeration §11.56 built now reckons as well as ranks: "how many
height facts do you hold?", "what is the total height?", "what is the
average height?" — aggregates over the same `_attr_family` scan
(extracted into a shared helper), governed by the same three honesty
rules: name the scope ("of the 3 height facts I hold"), name the
exclusions ("mill height (no number)"), and never let a denial touch a
number. Counting tells the whole truth in both directions: positive
beliefs are counted — word-valued facts included, a fact is a fact —
and denials are reported beside them as what they are ("I hold 4
height fact(s), plus 1 denial(s)"). Totals and averages convert
within §11.42's families to the largest unit present, say so aloud,
and carry the minimum confidence over every member. The §11.30
pairwise combine keeps its own door: "the sum of the tower height and
the bridge height" names its operands and is not an aggregate.

The gate (`experiments/aggregate_gate.py`) ran `_AGGREGATES_ON` off
against on, and between its two runs a latent harness hazard was
closed before it ever fired: the drift control demands exact
float-string equality, and different summation orders could differ in
the last bits — a spurious drift waiting on a seed. Cross-unit worlds
now draw meter values converting to exact binary eighths. The
authoritative run: **PASS, 0.343 -> 1.000, +0.657 at 5.05x seed sd,
zero drifted numbers** — no denied value moved any total or mean —
with the pairwise door and recall identical at 1.000 in both arms.

### 11.56 Superlatives: a claim about a set, with the set named

The comparative family's completion is setwise: "which height is the
tallest?" is a claim about EVERY height the library holds, so the
answer enumerates the attribute family through `fact_keys` — the
whole store, never a top-k sample that could silently miss the winner
— and names its scope: "The spire height is the tallest of the 3
height facts I hold: 2 kilometers." Negated facts are excluded by
polarity and accounted aloud ("1 denial(s) not counted — a denial
names no number"), non-numeric candidates are excluded BY NAME ("mill
height (no number)"), units convert where §11.42's table connects
them before any ranking, ties are named with every holder rather than
broken ("Tied for tallest ... barn height and spire height"), empty
families refuse, and confidence is the minimum over every included
candidate — the verdict rests on all of them. The enumeration is a
full scan by design: correctness before optimization, and the cost is
explicit where a sampled index would have been a silent wrong answer.

The gate (`experiments/superlative_gate.py`) ran `_SUPERLATIVES_ON`
off against on: winners in one unit and across units, downward
superlatives, ties, empty families (the baseline's free share, and
the variance), and worlds where the numeric maximum is stored NEGATED.
**PASS on the first run: 0.206 -> 1.000, +0.794 at 7.45x seed sd**,
zero denials crowned — the §11.48 line holding at set scale — and
recall at 1.000 in both arms. The comparative family is complete:
pairwise (§11.54), derived operands (§11.55), setwise (here) — every
verdict naming what it rests on.

### 11.55 Derived operands compare: three families in one answer

§11.54 registered its ceiling in the same breath as its capability:
the comparator compares RECALLED operands, and "is the mill hardness
bigger than the wall hardness?" — both sides derivable through
material chains, neither stored — refused honestly. This unit cashes
that ceiling the way §11.52 cashed §11.47's: an operand the store
does not hold is DERIVED through the chain machinery, under the full
discipline the question family already carries — §11.50's split rules
(a modifier-rescued derivation is a wrong split and refuses), §11.48's
polarity line (a derived denial holds no number to compare, and a
negated material derives nothing at all), §11.54's honesty extended
one step: the verdict marks WHICH side was derived and its trail
("Yes - mill hardness is 500 units (derived via steel), wall hardness
is 400 units (derived via iron)").

The gate (`experiments/compderive_gate.py`) ran `_COMPARE_DERIVES`
off against on: both-derived and mixed comparatives, ~25% broken-chain
comparatives as the unwinnable share, negated materials, with
stored-operand comparatives and recall as floors. **PASS on the first
run: 0.000 -> 0.674 at 5.84x seed sd**, every derived side trailed,
zero verdicts through a broken or denied chain, floors at 1.000 in
both arms — the flag gated only derivation, exactly as claimed. Three
families compose in one answer now: the chain machinery supplies the
operands, the §11.42 table converts them, the comparator delivers a
verdict checkable at a glance.

### 11.54 Comparatives: a verdict that was computed, refusals that are named

"is the tower taller than the bridge?" reached the polar machinery
through two defects at once. The derive path ran it through the
combine machinery — whose conclusion is None — so the claim never
matched and BOTH directions answered No: a coin that always says No.
And a missing operand fell to the direct matrix, which asserted "No -
tower height is 300 meters, not taller than citadel height" — a
verdict fabricated over an operand the library does not hold, the
absence-is-never-no line violated through the comparative door.

Comparatives now have an owner (`_polar_compare`, before the subject
ladder): the two operands are recalled, converted where §11.42's table
connects their units, and compared — the verdict naming BOTH values so
it can be checked at a glance ("Yes - tower height is 300 meters,
bridge height is 120 meters"), equal values answering "No - they are
equal", downward comparatives inverting. Four refusals, each naming
its reason: a missing operand ("I hold nothing for 'citadel height'"),
a negated operand (§11.48: a denial holds no number to compare), a
non-numeric operand, and cross-family units ("no definition connects
them"). The comparator compares RECALLED operands by design; derivable
operands refuse honestly, and comparing them is a registered
candidate.

The gate (`experiments/comparative_gate.py`) ran the flag's two arms.
Its first run measured the mechanism perfect and failed the variance
rule — the eighth "nothing varied" in the book — and was hardened the
standing way, the derived-operand ceiling scored into the metric at
~25%. Then: **PASS, 0.000 -> 0.893 at 12.91x seed sd, zero verdicts
without operands — while the coin arm fabricated 38 of them**, the
shipped defect measured at its full rate beside the fix, and recall at
1.000 in both arms.

### 11.53 A change of mind, said out loud

The episode log has recorded every revision since §11.16's era, truth
maintenance has retracted resting derivatives since §11.30, and §11.48
made polarity flips revisions — but the reply said the same bare
"Noted:" for a change of mind as for news. The honest-aloud principle
stopped exactly where a mind most owes an account: when it contradicts
itself. Now `remember_fact` reports its outcome (new / reinforced /
revised, with the old spoken form and the retracted keys — an additive
return nothing previously read), and the Learn stage narrates a real
conflict: "Noted: tower material is steel. That revises what I held:
tower material was iron. 1 derived fact(s) rested on it and were
retracted: 'tower hardness'." Polarity flips speak both sides ("was
not steel"). The line the controls hold at zero: only a REAL conflict
earns the notice — a reinforcement is not a revision, news is not a
revision, and a system that cries "revised" at agreements is as
unaccountable as one that revises silently.

The gate (`experiments/revision_gate.py`) ran `_REVISION_ALOUD` off
against on. Its first run measured the mechanism perfect and FAILED on
the variance rule — every world gave +1.000, the seventh "nothing
varied" in the book — and was hardened the standing way: the
unwinnable scored into the metric, never the rule softened. The
honest unwinnable is the exact-key ceiling: "the old tower material
is ..." against a stored "tower material" stores a NEW fact and
narrates nothing, by design. With ~25% near-key statements scored at
zero for both arms (and doubling as false-notice bait): **PASS,
0.000 -> 0.803 at 5.03x seed sd**, zero agreements called revisions,
zero near-key false notices, and semantics at 1.000 in both arms —
speaking the change changed nothing about what happened.

### 11.52 The fourth fact: a stated limit moved by measurement

§11.31 bounded the spread at two bridge hops with a warning — "longer
chains multiply pun risk faster than they add reach" — and §11.47
re-recorded the limit after measuring three-fact chains: the fourth
fact refused by the decay floor, "the architecture's stated limit, not
a promise". But the warning predates the rules that carry the pun
resistance now: single-origin lineages (§11.37), origin coverage,
order-as-evidence, whole-value bridging with accounted tokens
(§11.47), synchronous frontiers (§11.47). Whether it still bound was a
question for a measurement, and the depth-four gate
(`experiments/depth4_gate.py`) asked it: entity -> agent -> place ->
region -> attribute worlds (four facts, three bridges), ~25%
broken-middle chains as the unwinnable share, the full §11.47 decoy
family at the new depth, three-chains, two-chains, and recall as
regression floors — the candidate constants (`_MAX_HOPS = 3`,
`_FLOOR = 0.1`) monkeypatched per arm so the source would move only on
a pass.

**PASS on the first run: four-fact chains 0.000 -> 0.708 at 2.37x
seed sd, with ZERO decoy assertions at depth** — epithet agents and
ghosts refused through three bridges — and every shallower floor
identical in both arms. The warning was right when written: the pun
resistance lived in the decay floor then, and depth multiplied risk.
It lives in the structural rules now, and under them the third bridge
adds reach without adding puns. The constants shipped: a four-fact
chain's origin arrives at 0.125 over the 0.1 floor and answers with
its whole trail ("temperate, via wren, york, norland"); a fifth fact
arrives at 0.0625 and stays structurally refused — recorded the way
this limit was, a measurement away from moving, never a promise.

The full suite then surfaced the one downstream consequence: §11.33's
ladder economy pin. Depth-4 questions used to climb with exactly one
confirmation; they converge directly now, zero confirmations, and the
ladder's territory starts at depth 5 — where the one-confirmation
protocol holds exactly one rung higher (measured: a five-fact chain
closes with one consolidated rung, and revising the bottom fact takes
the rung down with it). The protocol did not break; it moved up with
the capability, which is what a protocol built on rungs is for.

### 11.51 Why: the trail is the answer, askable — and never invented

Since §11.31 the architecture's thesis has been that an answer IS its
evidence trail, and the provenance has been load-bearing since §11.30
(`derived_from` on every consolidated fact, premises on every
inference, confidence and reinforcement on every statement). But
nothing could ASK for it. "why is X Y?" now answers from provenance,
each kind in its own voice: a consolidated fact cites its premises
("Because tower material is iron; iron hardness is 490 units —
consolidated from a confirmed derivation"), a stated fact cites its
statement ("Because it was stated directly, reinforced N time(s)"), a
derivable-but-unstored subject derives and says so ("— derived just
now, not stored"), polarity included ("Because dome city is york; york
climate is not temperate").

The two lines that make the answers worth having are zero-tolerance:
**a false premise is corrected, never rationalised** ("why is the
tower material steel?" -> "It isn't - tower material is iron"), and
**an unheld subject is never explained**. Inventing a justification
for a premise the system does not believe is the transformer failure
mode this architecture defines itself against — an explanation machine
that will explain anything explains nothing. The subject ladder reuses
§11.50's split discipline (derivation tried at each size, stored hits
only with short claims, no trailing negators, modifier-rescued splits
refused).

The gate (`experiments/why_gate.py`) ran `_WHY_ANSWERS` off against
on: five kinds in world-varying mixes — stated, consolidated (derived
and affirmed through the live §11.33 protocol during world build),
derived, corrected, unheld — with recall as the floor. **PASS on the
first run: 0.170 -> 1.000, +0.830 at 19.88x seed sd**, zero false
premises rationalised, zero unheld subjects explained, recall 1.000 in
both arms. The silent arm's 0.170 is the unheld share its hedge
answers by being honest. Interrogative reach now spans what, is, and
why — recall, verdict, and provenance — all through one belief store.

### 11.50 Polar questions answered through chains: the family closes

The seam left between §11.48 and §11.49: "is the dome city climate
temperate?" when no fact holds the key "dome city climate" — derivable
("dome city is york"; "york climate is not temperate" entails No), but
the polar branch only consulted recall. Now a polar question whose
subject is not stored DERIVES it: the subject ladder walks longest
first, treating "subject" as a stored key OR a derivable one, and the
derived value meets the claim under the same matrix stored facts use —
polarity included ("No - the dome city climate is believed not
temperate, via york (derived)"), absence still never no, every verdict
marked derived with its trail, and an affirmation consolidating the
conclusion under §11.33's protocol.

Building the ladder caught two split bugs before the gate ran. A
stored SHORT prefix could claim a question about a longer derivable
subject — "dome city climate temperate" matched the stored "dome
city" and answered about the city — so derivation now gets its try at
each size before shorter stored keys are consulted. And a wrong split
could survive by modifier tolerance — "tower hardness 490" derived
with '490' read as a modifier — so a derivation that reads part of
its own subject away refuses (the dropped word belongs to the claim),
and a trailing "not" may never end a subject (the polarity word
belongs to the claim: "dome city climate not" / "temperate" once
answered No where the claim agreed).

The gate (`experiments/polarderive_gate.py`) ran `_POLAR_DERIVES`
off against on: derivable polar questions of four kinds, polar
questions over negated middles (the inhibition line crossing into the
interrogative form, zero tolerance), direct-fact polar and recall as
floors. **PASS on the first run: 0.688 -> 1.000, +0.312 at 5.11x seed
sd** — perfect on every derived kind, zero on every control, floors
at 1.000 in both arms. The §11.48 family is closed: statements parse
polarity, storage carries it, reasoning is inhibited by it, terminals
speak it, and questions — recalled or derived — meet it with one
matrix.

### 11.49 Denials speak at terminals, and only at terminals

§11.48 made negation inhibitory everywhere and deliberately left
negative knowledge silent at depth: "the dome city is york" plus "the
york climate is not temperate" refused, though together they honestly
entail "believed not temperate". This unit opens exactly one door — a
negated fact may be reached as a chain TERMINAL and answer, polarity
spoken and trail named ("the dome city climate is believed not
temperate, via york"), the premise line reading the denial as a denial
("york climate is not temperate") — and it still never seeds, never
relays, never bridges. Structurally: negated facts stay out of round-0
seeding, join the spread only as bridge targets, and a relay guard
stops every lineage at them; `_NEGATED_TERMINALS` is the gate's
baseline switch. Affirming a negative conclusion consolidates it WITH
its polarity, so a derived denial is inert as a bridge forever after,
exactly like a stated one — and negative knowledge composes with
§11.47's depth ("believed not temperate, via wren, york").

The gate (`experiments/negchain_gate.py`) ran door-closed against
door-open over shared worlds — negative-terminal chains (~25% with the
terminal never stored, the unwinnable share), negated-middle questions
as the §11.48 inertness control re-asserted under the open door,
positive chains and recalls as the floor. **PASS on the first run:
0.000 -> 0.694, +0.694 at 2.09x seed sd**, every delivered answer
named its polarity and trail, negated middles derived ZERO times in
either arm — the door measured at exactly its stated width — and
neither chains nor recall moved. The remaining rung of the §11.48
family (polar questions answered THROUGH chains: "is the dome city
climate temperate?" derived rather than recalled) stays registered.

### 11.48 Polarity: belief-of-absence, inhibitory everywhere, honest aloud

The library had no concept of negation, and the probe that opened the
unit caught the cost live: "the dome material is not steel" stored
"not steel" as a value, `_fold` dropped "not" as a stopword, and the
chain machinery derived the dome's melting point "via not steel" — a
denial bridged as an assertion, the premise trail displaying the exact
statement it contradicted. The curiosity path minted "not steel steel"
from the same record. Negation is now a first-class polarity:

* **Stored as identity.** `remember_fact(..., negated=True)` keeps the
  bare value with a flag, and polarity is part of what the fact IS — a
  positive restatement of a negated value is a REVISION (change of
  mind, episode logged, derivatives retracted), never a reinforcement.
* **Inhibitory everywhere the network reasons.** Negated facts neither
  seed nor relay the spread, never join arithmetic (a negated numeric
  summed its 120 anyway when first probed), and never mint curiosity
  premises. A denial says what may NOT be believed; letting it excite
  anything was the fabrication.
* **Honest aloud.** Every surface that speaks a value speaks its
  polarity ("dome material is not steel"), and statements parse it
  ("is not X" / "is never X") behind `_NEGATION_AWARE` — the gate's
  baseline switch.

With polarity comes the question form that needs it: **polar
questions**. "is X Y?" answers yes against matching belief; no against
contrary belief — a different stored value ("No - tower material is
iron, not steel") or a stored negation ("No - believed not steel");
inverts cleanly for negated claims ("is X not Y?"); and holds the line
the section is named for: a negation of one value says nothing about
another ("is the dome material iron?" -> "I don't know - I hold only
that dome material is not steel"), and **absence is never no** — an
unheld subject hedges, because not believing X and believing not-X are
different states of mind, and collapsing them is how a system starts
denying things it merely hasn't heard of.

The gate (`experiments/negation_gate.py`) ran the flag's two arms over
shared worlds: polar questions of four kinds in world-varying mixes,
entities whose only material knowledge is negative, unheld subjects,
genuine chains and recalls as the regression floor. **PASS on the
first run: polar 0.273 -> 1.000, +0.727 at 28.33x seed sd** — the
margin wide because the capability simply does not exist in the
baseline (its 0.27 is the unknown-kind share its hedge answers by
being honest). The baseline arm derived a chain straight through a
denial **36 times** — the old fabrication measured at its full rate
beside the fix — while the polarity arm derived zero, denied zero
unheld subjects, and moved neither chains (1.000) nor recall (1.000).
Negative chain answers at depth ("believed not temperate", derived)
stay a registered candidate: phase one made denials inert; making them
speak is its own claim or none.

### 11.47 Depth: three-fact chains, and the three layers the gate indicted

Three-fact chains — two bridges, "the tower architect is wren; wren's
birthplace is york; york's climate is temperate" — turned out to be
mechanically reachable inside `_MAX_HOPS = 2`: the middle fact's
direct contact merges into the carried lineage, and decay 1.0 -> 0.25
clears the 0.2 floor. Found is not believed: no gate had ever measured
depth, and a four-fact clean-room probe immediately caught a
depth-specific pun. The depth gate (`experiments/depth_gate.py`) was
pre-registered — entity -> agent -> place -> attribute worlds, ~25%
broken-middle chains as the unwinnable share, epithet decoys ("wren
the younger" holds the role, only plain wren holds a birthplace),
ghost entities, a one-hop baseline arm — and its three runs each
indicted a different layer:

* **The exact branch asserted sub-keys.** Run one scored 3-chains at
  0.000 in BOTH arms with 72/72 decoys "asserted" — every one the same
  interception: Recall's candidate ladder tries sub-n-grams, the
  question surface's exact branch believed "hit" meant "exact", and
  "what is the high mill sculptor workshop population?" was told who
  the sculptor IS while the chain inference that could answer what was
  ASKED sat unreached below. §11.29's coverage rule now guards the
  exact branch too: assert only when the key covers the question,
  else fall through.
* **Hop depth depended on dict order.** Run two's "one-hop" arm
  answered 3-chains at 0.667 — impossible if `_MAX_HOPS` meant depth.
  Origins were mutated in place during a round's pass, so a lucky
  iteration order cascaded origin -> middle -> target inside one
  "hop". Cognition must not depend on dict order: rounds are now
  synchronous frontiers (updates collect during the pass and apply
  after it), and the hop cap finally means what it says.
* **A bridge carried on any shared token.** The same run measured the
  epithet pun genuinely: 72/72 asserted through inference — plain
  wren's birthplace, york's climate, delivered for "wren the younger",
  the premise trail displaying the substitution it had just made. The
  whole-value rule closed it in both directions: the value's tokens
  must all appear in the target key ("wren the younger" may not hop
  through "wren birthplace"), and the target's unaccounted tokens must
  be asked-for — except ONE in terminal position, the relation slot
  ("bronzite BASE" traverses), origin coverage's own allowance applied
  to every link. An unaccounted token INSIDE the subject ("wren the
  YOUNGER birthplace" receiving plain wren) is a qualifier being
  dropped, and dropping qualifiers is how one entity substitutes for
  another.

Run three, all layers honest: **one-hop 0.000, two-hop 0.688, +0.688
at 2.85x seed sd**, zero assertions on either decoy form in either
arm, recall 1.000 in both. The two-hop miss is the broken-middle share
scoring zero by design; the one-hop zero is the baseline finally
meaning depth. The world builder needed its own fix on the way — an
epithet decoy's backing fact could silently re-link a "broken" chain,
so its refusal expectation was scoring a lie until forbidden
(agent, relation) pairs kept broken chains broken — harnesses must be
indictable too. The fourth fact stays refused by the decay floor: the
architecture's stated limit, not a promise.

### 11.46 Two adjectives may decorate: the cap moves, the floor stays

§11.36 shipped adjective tolerance at exactly one dropped token and
recorded its honest ceiling: double-modifier questions refused at
0.333, "because two unknown tokens is no longer a decoration". That
ceiling was a registered candidate, and this unit moved it — the
inference path now drops at most TWO library-unknown tokens
(`_MAX_MODIFIERS`), and the safety line moved with it. The single-drop
guard ("len >= 3") generalises to a **residual floor**: at least two
known informative tokens must survive the drop, so "the old weathered
tower conductivity" reads both adjectives away and converges through
`tower material -> steel`, while "the ancient obelisk density" thins
to one token and refuses — a question that is mostly unknown is not
decorated, it is not understood. Convergence still demands a bridge
(the §11.35 structural argument, unchanged), every answer names every
word it read away in question order ("reading 'crumbling' and
'weathered' as modifiers") so the reading stays vetoable, and the
single-drop wording is untouched. The curiosity path's positional
droppable rule stays at one: its unknown-counting is structurally
different (the asked attribute is itself library-unknown there), and
extending it is its own claim or none.

The gate (`experiments/multimodifier_gate.py`) ran the shipped cap-1
as its baseline arm against cap-2, all claim-worlds double-modifier,
~25% triple-modifier worlds as the pre-registered unwinnable share,
and two decoys in every world. **Its first run indicted the harness,
not the mechanism**: 24/24 decoys "asserted" in BOTH arms — including
the baseline running shipped behavior that measured zero in §11.36 —
because the check counted any response containing the value, and
§11.29's demote hedge ("I don't hold that exactly. Nearest I hold:
…") names the value under its own key without asserting it for the
ghost. The registered word is "asserted"; the harness was corrected to
§11.36's operationalisation of it, worlds and seeds untouched. Then:
**baseline 0.000, extended 0.583, +0.583 at 1.13x seed sd — a narrow
pass, declared narrowly** in §11.13's tradition, and narrow for an
honest reason: the RNG dealt 5/12 triple worlds (42% against the
intended ~25%), each scoring zero in both arms by design. Zero decoy
assertions of either form, every answer named both words. The
triple-modifier ceiling is recorded the way §11.36 recorded its own:
the next rung is a registered candidate, not a promise.

### 11.45 The VRAM tier: the storage split completes, at parity first

The vision's storage split — disk as the permanent library, DRAM as
the working set, VRAM as the hot layer — had two measured tiers and a
missing third. The DLL now holds resident ternary layers
(`uqg_layer_upload/forward/free/resident_bytes` in uq_cuda.cu): weights
cross the bus once, every later query moves only activations, and a
byte-budgeted LRU cache (`native/vram.py`) decides residency with
structural fallback — no GPU, no slot, refused upload all return None
and the tier below runs, bit-identical.

**The gate caught one cache bug and then PASSED on an RTX 4090**: an
oversized layer was becoming resident over budget (eviction empties
the cache and then has nothing left to give) — it now refuses, and the
fallback is the mechanism, not an apology. Then: parity **3.6e-15**
against the pure-Python reference, per-query 0.061 ms (per-call
upload) falling to **0.028 ms resident** (+0.033 ms at 3.26x
repetition sd), 2.7x over pure Python, ledger and device agreeing at
every step, residency zero after clear.

Rule 1 held the whole way — parity first, the clock second — and the
architecture's name is now measured at all three tiers: the library on
disk (§6), the working set in DRAM (§11.14), the hot layers in VRAM.

**And it runs live**: sessions build the cache wherever a CUDA device
exists (bit-exact parity is what justifies default-on), the expert
pool's predict path tries residency first and falls through
transparently, retraining an expert takes its resident layers down
with it (`drop_prefix` — a device copy is valid only while the expert
is frozen), and ':resident' reports the third tier beside the other
two. A pasted glyph's second recognition in a session answers from
VRAM.

The full suite then caught what the gate could not: the slot table is a
process-wide resource (64 slots in the DLL), and a session dying
without cleanup left its layers resident as orphans. Hundreds of test
sessions later the table was full, upload returned -1 for whoever asked
next, and the hardware parity tests — which had passed in isolation —
failed in company. A cache now returns its slots when it dies
(`__del__` → `clear`), reproduced at 70 dead caches: 1,408 orphaned
bytes and slot -1 before the fix, zero bytes and slot 0 after. The
accounting discipline's second catch in one section, and this one
needed the whole suite to surface — no gate builds hundreds of
sessions; the test corpus itself is the only world dense enough.

### 11.44 Plasticity: attestation earns rank, and nothing more

Rule 3 — "recall reinforces: the catalog reorganises around what is
used" — reached experts and the router long ago and never reached fact
retrieval: equal-overlap ranking ties broke by the alphabet. §11.30's
last registered candidate lands in its honest first form: a fact's
reinforcement count plus confidence breaks ties in BOTH search paths
(sharded and inline), so a re-attested fact surfaces above token-tied
siblings under top-k pressure. A tie-break only: reinforcement never
outranks a better token match, and the coverage rules upstream still
refuse whatever retrieval surfaces.

**The gate PASSED — narrowly on margin, absolutely on the controls,
with one accidental validation**: the first world build buried the
target under OTHER entities and both arms answered anyway, because
§11.38's phrase probes already rescue cross-subject burial — that fix
validating itself from an unexpected direction. The burial that
matters is within one subject (ten same-entity facts sorting ahead of
"material"), and there: chains 0.000 -> **0.571 at 1.11x seed sd**,
zero entrenchment falls (a five-times-reinforced sibling never bought
a ghost an answer), better token matches winning identically in both
arms. The 0.429 miss is the unreinforced worlds, where the alphabet
stands for both arms: attestation earns rank, absence earns nothing,
which is the entire claim.

### 11.43 Self-study: the loop that closes its own gaps, shipped on a zero

The pieces existed separately; `interpreter/study.py` is the loop — one
pass that surveys the gap queue, asks the panel (one model per
independent voice), quarantines everything, applies only what
independent voices corroborate through §11.41's rules, and reports
every application for veto. ':study [n]' runs it from the chat CLI.

The gate's whole point was the new risk: **hallucination through
corroboration** — independent lineages confidently completing the same
invented thing. Worlds seeded the survey itself (a term mentioned until
the gap survey minted a question about it): factual terms, contested
concepts, and invented words. Across two live passes, **zero invented
terms were corroborated** — "flurbostat" stayed open every time — zero
applications landed on a wrong subject, and closure ran 0.667 at 1.29x
seed sd with the two misses being contested concepts ("beauty",
"justice"-class) staying open: the loop declining to flatten a
contested idea into one voice's take, scored as misses on purpose.

The pipeline the library now trains itself through, end to end and all
measured: gaps found by usage, questions asked by curiosity or survey,
answers quarantined, corroboration counted by voice with containment's
bounded rules, application gated on acceptance, everything reported,
truth maintenance standing behind every stored fact.

### 11.42 Unit conversion: the refusal narrowed, nothing widened

§11.30 refused every cross-unit combination and registered conversion
as future machinery. The machinery is a definition table — three small
families (length, mass, time), every factor exact, conversion only
within a family, results expressed in the larger unit and labelled
"(units converted)" with both premises in the trail. What the table
cannot connect refuses exactly as before: degrees beside meters,
florins beside anything, and a bare number beside a united value —
assuming a missing unit would be invention.

**The gate PASSED**: cross-unit combinations answer correctly
(numerically checked against the exact factors) in 0.643 of all worlds
against the baseline's structural zero, **+0.643 at 1.29x seed sd**,
zero refusal falls in either arm, same-unit behavior byte-identical.
The 0.357 miss is the bare-number worlds, in the metric per standing
policy — refusing to assume is the mechanism working. One power
correction recorded: the first draw put bare worlds at half the
metric, where a Bernoulli contrast cannot clear one sd whatever the
mechanism does (§11.35's arithmetic).

### 11.41 Containment corroboration: the bounded exception, and the first learned facts

§11.40 registered the question its honest zero raised: three of four
voices' positions contained one another, and the consensus machinery's
own line — overlap is reported, "never credited" — met a case where
refusing to credit looked like the error. The gate decided it. The
mechanism (`_containment_core`): a core is an EXISTING position whose
tokens are a subset of its backers'; no word is ever equated with a
different word; a backing position carrying any negation token backs
nothing; the credited answer is the core — the least claim all merged
voices back — and the CONTAINMENT flag that turns it all off again is
one constant.

**The gate PASSED — narrowly on margin, absolutely on the controls**:
elaboration-world corroboration 0.000 -> 0.600 at 1.16x seed sd, with
**zero negation falls** ("earth round hoax" never backs "earth round"),
**zero paraphrase falls** (different words never merge), and exact
behavior identical between arms. The 0.400 miss is
overlap-without-nesting — the live run's "evolves through usage"
beside "system communication" — containment's honest ceiling, and what
keeps it orthography-class. The module's "never credited" is amended
to "credited only through this gate's rules".

**The re-run then taught the library its first panel facts.**
'language is communication' — three independent voices through the
containment core, applied through the normal answer path, recalling
live at 0.70. The run also caught its own driver counting corroboration
as application while a handler silently refused a bare-word answer
(fixed: application is `outcome.accepted`, and untrained-category
answers are formatted as statements). And it stored one flagged fact
for the human's veto: 'pixels' = '25' — grounded in this library's own
usage (the 5x5 grid's 25 cells) and corroborated under the rules, but
a count wearing a definition; one revision statement replaces it, and
truth maintenance handles the rest.

### 11.40 The first library-training run, three panel defects, and an honest zero

"Continue training" with the voice queue exhausted means the mature
channel: the library's own learn-queue questions put to the panel, one
model per independent voice, answers quarantined, only corroboration
applied. The first run returned zero corroborations — and its
transcript convicted three defects rather than the design:

1. **The panel path never passed the reasoning remedy.** gemma spent
   all 400 completion tokens thinking on every question; the
   distillation path had always sent ``reasoning_effort="none"``, and
   ``TeacherPanel.ask`` had not.
2. **Template tokens read as words.** command-r's
   ``<|END_OF_TURN_TOKEN|>`` normalised into the position suffix "end
   turn token" — the §11.19 leak in a second pipeline — so "earth round
   end turn token" could never exact-match another voice's "earth
   round".
3. **List markers read as positions.** "1. Paramiko is a python ssh
   library" split at the first ". " and filed the position "1".

All three fixed and pinned (`tests/test_lmstudio.py`). The re-run's
zero is the honest kind: four voices give four DIFFERENT true
elaborations ("system communication using symbols" / "communication" /
"evolves through usage" / "system communication"), and exact-match
corroboration refuses to equate them — the docstring's own line, that
judging two different words to mean the same thing would manufacture
consensus. ~50 claims sit quarantined for independent evidence (LLM
sources are excluded from corroborating each other in the stash by
§11.12's rule, so they wait for the web or the human).

The near-agreement pattern registers the successor question:
**containment corroboration** — three of four positions above contain
"communication", and counting token-subset cores as agreement might be
orthography-class rather than meaning-class. It needs its own gate
with negation decoys before it touches consensus ("the earth round
story is a hoax" contains "earth round" and denies it), and until that
gate runs, the bar stays exact.

### 11.39 The suggester goes live: the boundary translates, the core still refuses

§11.37's remedy, wired and adopted. When the lexical core cannot assert
on a question, the embedding suggester (`reason/semantic.py`) proposes a
reading over the library's OWN candidate set — the same phrase probes
the spread uses, a few dozen keys embedded in one batched call — and
the lexical core verifies twice before a word is said: the measured
cosine floor (0.75; §11.37 put 0.833 fabrication at 0.65), and §11.36's
positional rule made transitive — every question token the key does not
cover must precede a covered one, because a synonym modifies what
follows it and a trailing unknown is a subject that is never read away.
The reply names both the reading and the match ("Reading that as
'tower height': ... embedding match 0.88") so the reader can veto it.
Sessions default OFF (tests and gates stay deterministic without LM
Studio); the chat, GUI and TUI enable it from
``lmstudio.semantic_suggest``; absence costs one cached availability
probe a minute.

**The adoption gate caught one more fabrication shape before passing**:
"what is the HEIGHT of the obelisk?" anchors on the attribute word
alone and attribute-plus-attribute cosine clears any floor — the
wrong-metal fabrication wearing an embedding; the positional rule is
what killed it. Then, live-pipeline ON vs OFF:

| arm | synonym asserted | canonical+reorder | decoy falls |
|---|---:|---:|---:|
| off | 0.000 | 1.000 | 0.000 |
| on | **0.833** | 1.000 | 0.000 |

**+0.833 at 2.04x seed sd**, zero decoy falls in either arm, the
lexical families byte-identical between arms. The 0.167 miss is the
chained-synonym ceiling, included in the metric at ~35% of worlds on
purpose: it is the half §11.37 measured the embedding cannot buy,
because the answer is not a held key — that half belongs to the
spread, and the division of labor now runs live: **the boundary
translates, the core chains and refuses.**

### 11.38 The density gate: convergence survives three orders of magnitude

The make-or-break measurement from the "could this whole system work"
assessment. Every cognitive gate had run on five-to-ten-fact worlds;
this one built colliding-token worlds — multi-word entities from small
word pools, shared attributes, the real shard-backed store — and swept
100 / 1,000 / 10,000 facts with pre-registered bars: fabrication must
not grow with density, chains and recall must hold, latency reported.

**It failed four ways first, and every failure became architecture**
(the full saga in `experiments/density_gate.py`): subject consistency
(bridged evidence traces to one origin — no pooling puns from unrelated
subjects, which measured 0.500 fabrication at just 100 facts); origin
coverage (at most one origin token the question never said — no "low
mill" answering for a ghost "royal mill"); subject-prefix bucketing
with addressed retrieval (the true wall was the STORE: ranked bucket
selection collapsed into ties at density, and a subject's facts now
co-locate in a bucket its name computes directly, O(1) at any density,
legacy fallback for old stores); and order as evidence (a multi-word
origin must share an adjacent bigram with the question — no anagram
"barn spire" answering for a ghost "spire barn"; tied anagrams refuse).

**Then it passed clean**: chains 1.000, fabrication 0.000, recall 1.000
at every tier; latency 13 -> 59 -> 199 ms/query, sub-linear, and
entirely in the ranked path the addressed path bypasses.

### 11.37 The paraphrase gate: the language wall measured, the hybrid tradeoff quantified

The second make-or-break measurement. Four surface families over the
same intents, decoys in every family: canonical 1.000, **reorder 1.000**
(order-blindness confirmed), **verbose 1.000** (politeness filler is
free), **synonym 0.000** — the wall at full height by the assertion
standard, with its practical shape recorded: "how tall is the tower"
still DELIVERS the right number hedged ("Nearest I hold: tower height
is 417 meters"), while the chained synonym misses entirely and leaks a
junky though hygiene-passing curiosity. Zero fabrications in every
lexical family.

**The embedding comparison** (text-embedding-nomic): at cosine>0.75 it
buys 0.500 of the synonym wall with zero decoy falls; at 0.65 it
fabricates **0.833** of the ghosts. The threshold is not a tuning
detail — it is the entire safety argument, and the half the embedding
buys is exactly the direct-attribute half: the chained half needs the
library's spread, which key-matching cannot do. That is the hybrid
division of labor, measured: embeddings translate surface forms at the
boundary, the library chains and refuses at the core, and neither can
do the other's job.

### 11.36 Adjective tolerance: one word may decorate, never substitute

§11.35's registered claim, closed with the safety argument carried in
the structure. One library-unknown token may be dropped and the spread
retried — on the inference path ONLY. It is safe there because
convergence still demands a bridge: drop "tungsten" from "the melting
point of tungsten" and the residual's coverage is all-direct, which the
spread already refuses as plain recall's territory; drop "weathered"
from "the weathered tower conductivity" and the residual converges
through ``tower material -> steel``. The recall path never gets the
drop — there, the same trick would assert steel's number for tungsten,
the §11.29 fabrication verbatim. Two unknown tokens refuse outright:
that is not decoration, it is a question the library does not
understand, and saying so beats guessing which word mattered. Curiosity
gets the tolerance too, but by a POSITIONAL rule, because there the
one-unknown test cannot work: in the missing-premise scenario the asked
attribute is itself library-unknown (that is why it is missing), so
"weathered tower conductivity" holds two unknowns. An adjective precedes
a key-known token; the asked attribute precedes nothing known — so
"weathered" drops, "conductivity" survives into the ask, and the ghost
in "obelisk melting point" drops the same way only to die at the
empty-remainder guard.

**The gate PASSED:**

| arm | tolerated answers |
|---|---:|
| baseline | 0.000 |
| modifier | **0.667** |

**+0.667 at 1.35x seed sd**, zero unknown-entity fabrications across
every decoy in either arm, every tolerated answer naming the token it
dropped ("reading 'weathered' as a modifier") so a reader can veto the
reading. The 0.333 miss is the double-modifier worlds, refused by the
at-most-one rule. Tolerance without a new hole, because the old guard
does the guarding.

### 11.35 Compound questions: sequential attention, honest halves, hygienic asks

"What is the tower melting point and the bridge length?" used to demote
to one nearest-held fact and mint a malformed curiosity — "steel melting
point bridge length", a clause wearing a fact key. The decomposition
path splits a genuine conjunction and runs each part through the FULL
single-question machinery — exact recall, the spread, an honest unknown
with its own well-formed ask — then composes the reply: recall beside
inference beside a named gap. A half-known compound answers the half it
holds and says which half it does not, and the unknown part's curiosity
is the PART's ("steel conductivity"), not the clause's.

The decoy that matters is arithmetic: "the sum of A and B" contains
" and " and is one question about two facts, so combination words veto
the split — and now veto curiosity minting too, killing the second junk
key the probe exposed ("steel sum height bridge height").

**The gate PASSED — after breaking its own decoy twice** (a reused
entity made a third fact match the combine's content, so the exactly-two
rule refused every decoy in BOTH arms: a broken decoy, not a held one;
and unknown-part worlds were double-counted as dilution when their own
criterion already owned them; the adjective share was then set by power
arithmetic, because a Bernoulli contrast at 50% share cannot clear one
sd whatever the mechanism does):

| arm | full answers | honest halves | combines held |
|---|---:|---:|---:|
| baseline | 0.000 | 0.000 | 1.000 |
| compound | **0.778** | **1.000** | 1.000 |

**+0.778 at 1.76x seed sd**, zero malformed curiosity keys. The 0.222
miss is the adjective worlds — "the weathered tower conductivity"
defeats the coverage rules that keep pun-rejection honest — recorded as
a real limit: the next representational claim on this path is adjective
tolerance, and it will need its own gate.

### 11.34 The ladder: the system's own hints steer its teaching, at exactly minimal cost

§11.31 left "stepwise confirmation of longer chains" unmeasured, and
building the protocol found the better claim: the curiosity hints
(§11.33) already form the guide. A refused deep question names the
premise it needs; asking THAT question either derives (two bridges of
reach) or refuses with the next hint down; each confirmed derivation
consolidates two bridges into one recallable fact. The gate's driver is
deliberately mechanical — follow the hint, say "yes" to what derives,
know nothing about the world — because the claim is that the SYSTEM
carries the plan, not the teacher.

Two protocol corrections first, both conservative, both recorded:
depth-3 worlds sat inside §11.30's direct reach and closed in both arms
(+0.500 at 0.94x sd of pure dilution), and the driver was affirming the
deep question itself, padding the confirmation count. Then:

| arm | closure | stale after revise |
|---|---:|---:|
| baseline | 0.000 | 0.000 |
| ladder | **0.625** | 0.000 |

**+0.625 at 1.21x seed sd** — §11.31's narrow-margin class, from the
same binary world structure — and the finding worth the section:
**1.80 confirmations per closed world against a depth-minimal 1.80**.
Hint-guided climbing is not merely sufficient, it is exactly minimal.
Broken-rung worlds all stalled cleanly; after revising the chain's
bottom fact, zero stale assertions — the recursive retraction takes the
whole ladder down with its premise.

What the cognitive loop now composes to, measured at every joint:
a question too deep to answer guides its own teaching down to the
reachable rung, climbs back up on confirmations, answers, survives
revision honestly — and every step of that sentence has a gate behind
it (§11.30, §11.31, §11.33, here).

### 11.33 Curiosity: a refused inference asks for exactly what it was missing

"Teach it so that it can learn." The learn queue could survey for
statistical gaps — terms seen often, categories without experts — but a
refused inference was a dead end, even though the partial activations
knew precisely which premise was missing. `missing_premise` turns that
knowledge into the ask: the held bridge's value plus the uncovered
remainder. Asked "what is the tower conductivity?" while holding only
``tower material -> steel``, the refusal now says "If I knew the steel
conductivity, I could work this out — ':learn' will ask", the learn
queue ranks that question first (a gap a USER's question actually hit
outranks every surveyed guess), and the prompt shows the whole chain so
the human sees why it is asked. Answering it drains the curiosity and
the re-asked question converges through what was just taught. The loop:
**fail -> ask -> learn -> infer.**

The failure mode is question spam — §11.16's noise wearing a curious
face — and the guard is groundedness: curiosity flows only forward along
a held bridge whose value names a thing (non-numeric, at most two
informative tokens). "What is the obelisk melting point?" touches
``steel melting point`` through its attribute tokens, but 1370 degrees
is a quantity, not a subject, and no question is generated.

**The gate** (live pipeline + real LearningSession, baseline arm with
`missing_premise` disabled): its first run measured +1.000 at sd 0.000
and was refused by the zero-variance rule — the fourth deterministic
mechanism this session to ace a protocol built only of worlds it could
win. The conservative correction, again: score the unwinnable worlds
(~30% hold no bridge at all) as part of closure for both arms. Then:

| arm | loop closed | ghost spam | ungrounded asks |
|---|---:|---:|---:|
| baseline | 0.000 | 0.000 | 0.000 |
| curiosity | **0.750** | 0.000 | 0.000 |

**+0.750 at 1.62x seed sd, premise precision 1.000**, zero spam, zero
asks where nothing held pointed anywhere. The 0.250 miss is the
no-bridge worlds, and that is the honest reading of the mechanism:
curiosity cannot outrun the library's groundedness — it converts one
held bridge plus one human answer into one new reach, which is exactly
what "teach it so that it can learn" asks the machinery to do.

### 11.32 The validator cannot be tightened into the reader, and the shelf gains a resident

§11.28 registered one open mechanism: tighten the shift-Jaccard validator
against the curation file as ground truth. Built as
`strict_contradiction` — the historical nearest-anchor rule plus the two
checks the reader's rejections decompose onto, a floor (18 of 31 read as
*nothing*) and a margin (13 read as a *different* label). The gate split
calibration from evaluation (tuning thresholds on the data that grades
them manufactures a pass): per split seed, grid-search (margin, floor) on
half the verdicts, score the untouched half. Bars pre-registered: keep
>=95% of reader-kept, catch >=50% of reader-rejected.

**It FAILED, decisively**: held-half recall **0.050**, keep rate 0.947 —
under its own bar, so even five percent recall was bought by shedding
good variants — and the calibration never stabilised (two of five splits
found no admissible grid point at all). The rejected items do not sit
lower in shift-Jaccard agreement than the kept ones. The reader rejects
on **shape gestalt** — whether a taper points, whether a silhouette reads
— which agreement-under-shift does not encode on any threshold.

Two consequences. The blind-reading protocol (§11.28) is **permanent**,
not stopgap: mechanical validation catches gross errors at drawing time,
and admission to the training bank goes through a reader. And
`strict_contradiction` becomes the built-but-not-adopted shelf's third
resident, beside `center_glyph` (§11.21) and `scale_normalize` (§11.22):
correct at what it does, tested, measured out of a job — the shelf itself
now a small monument to the difference between building a mechanism and
earning its deployment.

### 11.31 Consolidation: a confirmed thought becomes memory, and revision takes it back

§11.30 ended with consolidation registered — the episodic-to-semantic
move, and the second half of the brain-shaped story: derivation that
earns confirmation should stop being re-derived and become recallable,
usable as a premise. The signal question decides everything here.
Repetition is NOT a signal — §11.16 measured reinforcement without
confirmation making errors 3.9x harder to overturn — so the only
promoter is the user's explicit affirmation, and pending state survives
exactly one turn: "yes" immediately after a derivation consolidates it;
"yes" after anything else agrees with something else.

The price of materialising a conclusion is staleness, so **truth
maintenance** ships in the same commit: a consolidated fact carries its
premises (`derived_from`), and revising any premise retracts every
conclusion resting on it, recursively, with a retraction episode logged.
The staleness control caught a real defect before the gate even ran: the
fact-shard store's `flush` skipped emptied buckets, so a retracted fact
resurrected from the stale vault bucket — and recall-reinforcement then
re-persisted the ghost. Deleting the last fact in a bucket was a silent
no-op. Fixed in `factshards.flush`, pinned in
`tests/test_consolidation.py`.

**The gate** (two arms over generated depth-3 worlds, baseline with
`consolidate_fact` disabled): derive the intermediate step, affirm, then
ask the question §11.30 measured as refused. It forced two metric
corrections first — the honest "Nearest I hold" demotion was being
counted as an assertion because the value string appears in it, and the
first world build miscounted its own hops — then **PASSED, narrowly, in
§11.13's tradition of saying so**:

| arm | deep reach | stale rate | repetition leak |
|---|---:|---:|---:|
| baseline | 0.000 | 0.000 | 0.000 |
| consolidation | **0.625** | 0.000 | 0.000 |

+0.625 at **1.21x seed sd**, zero stale answers after premise revision,
zero repetition leaks. The 0.375 miss is the depth-4 worlds, where the
intermediate itself sits past the activation floor and nothing earns
confirmation: **one confirmation buys exactly one step of reach**.
Stepwise confirmation of longer chains is unmeasured and would need its
own protocol. With this, the cycle the architecture claims is closed and
measured end to end: perceive -> recall -> infer -> confirm ->
consolidate -> revise -> retract.

### 11.30 Inference by spreading activation: the library thinks, and the gate holds

The question path could recall what it was told and could derive when
*commanded* (``goal:``). What it could not do was notice, on its own, that
an answer it was never told exists implicitly in what it holds. That gap is
also where this architecture makes its differentiating claim, so the
mechanism is deliberately **brain-shaped rather than transformer-shaped**,
and the differences are structural:

* **Activation, not attention.** A transformer infers in one dense forward
  pass — every parameter participates, and nothing about the answer says
  which memories produced it. Here a question injects activation at its
  concept tokens; activation spreads along stored associations (a fact's
  key tokens to its value tokens, and value tokens are other facts' keys);
  only touched facts ever page in. Inference has the same sparse-recall
  economics as the shard library itself — nothing is resident because
  everything might be needed.
* **Convergence, not completion.** An answer forms where activations from
  *different parts of the question* meet. "what is the tower melting
  point?" spreads from {tower} and {melting, point}; ``steel melting
  point`` is reached from both — via ``tower material → steel`` on one
  side, directly on the other — and that convergence is the answer. With
  iron's melting point also held, iron is never touched by the tower half
  and stays silent: **a fact reached from only part of the question is a
  pun, not a proof**. Every fabrication in the book (§11.14's stopword
  answer, the tungsten question, §11.29's wrong-metal assertion) is an
  activation spread without this rule.
* **The trail is the answer.** Every inference names the facts it crossed,
  is marked "inferred, not stored", and carries the **minimum** confidence
  along the path — derivation dilutes belief, never concentrates it.
  Arithmetic combination (sum/difference/larger over two named numeric
  facts) sits beside the spread, and mismatched units refuse: 300 meters
  plus 2 kilometers is not 302 anything.
* **Bounded honesty.** Activation decays 0.5 per bridge over a 0.2 floor:
  two bridges reach (0.25), three stop (0.125). The floor is stated, and
  the gate measures it instead of hiding it.

**The gate** (`experiments/inference_gate.py`) runs the full live pipeline
in both arms — baseline disables only the inference call — over generated
fact worlds with four question sets: direct recall, chains (depth 1–3),
combinations, and decoys that share tokens with held facts but connect to
nothing. It failed its own zero-variance rule twice first — a
combine-selection defect made every world +0.500 exactly, then too-easy
worlds made every world +1.000 exactly, and at sd 0.000 any margin is
unfalsifiable. The worlds were hardened, not the rule. Then:

| arm | direct | inferred | decoy falls |
|---|---:|---:|---:|
| baseline | 1.000 | 0.000 | 0.000 |
| inference | 1.000 | **0.938** | 0.000 |

**+0.938 at 8.10x seed sd**, zero decoys fell in either arm (cross-unit
sums included), recall untouched. The 0.062 miss is entirely the depth-3
worlds — the designed floor, measured. `tests/test_inference.py` pins the
mechanism, the guards (each named for the fabrication it prevents), and —
§11.11's discipline — proves the decoy control *can* fail: an eager
matcher reaches every decoy the convergence rule refuses.

What this deliberately defers, each a registered candidate for its own
gate: **consolidation** (an inference that keeps being used should be able
to earn promotion toward a stored fact — through §11.17's
confirmation gate, because §11.16 measured what unconfirmed reinforcement
does), unit conversion, and activation with learned edge weights.

### 11.29 A battery against the live chat surface, and six defects fixed

Component tests green is not the surface working — §11.5's lesson, aimed
this time at the chat CLI. A scripted battery drove every documented
interaction through the real `--script` surface, and six defects fell out,
each now fixed with the regression that would have caught it
(`tests/test_chat_surface.py`):

1. **A pasted glyph answered row by row.** The README's first example —
   paste five glyph rows — arrived as five separate inputs, each told "I
   have nothing on that". The CLI now collects a bare glyph row into the
   full grid (blank line escapes, a non-glyph line is processed normally,
   EOF ends collection) and the paste reads as one glyph again.
2. **Total ignorance claimed over a stocked quarantine.** "what is code?"
   answered "I don't hold anything on that" while the stash held staged
   claims about code. The unknown reply now counts staged claims sharing an
   informative token with the question and points at ':stash'/':analyze'/
   ':promote' — pointing, never quoting, because staged text has not earned
   belief.
3. **The plural fold stopped at the router.** "what are towers?" missed the
   just-stored `tower height` because the question fallback compared raw
   tokens. Both sides now fold through `normalize_token`, and the folded
   token set is probed against the fact index as well.
4. **The planner failed without naming the missing part.** "no plan reaches
   the goal within depth 6" is search internals, not an explanation — and a
   goal naming one held and one unheld thing is *structurally* unplannable
   (derived-from-all needs every input). The failure now names what is
   missing and both remedies; a plan that succeeds while ignoring an unheld
   term now says so.
5. **The full test suite wrote the developer's real settings file.** TUI
   tests never isolated `ULTRAQUANT_CONFIG`, and one suite run left a
   deleted temp directory as the standing session home. Tests now isolate;
   the TUI and GUI both refuse to persist a home under the system temp
   directory at all, because an ephemeral session must never become the
   default.
6. **An empty training run printed a zero table.** With every voice already
   taught, `--next` reported `held-out routing 0.000 -> 0.000` — a broken
   router's numbers for a run that never happened. The empty report now
   omits the measurement block.

The battery also confirmed two behaviours worth keeping on record: the
tungsten fabrication guard still refuses to invent, and the voice-queue
"already taught" verdict was checked against live lineage data and is
correct — cydonia and the qwens are siblings of used teachers, so the
phrasings channel stays at its measured limit until a genuinely new voice
appears.

### 11.28 The blind reader confirmed the last suspect, and the chain ends with a mechanism

§11.27 said the remaining question needed an oracle. The user delegated the
reading, so the oracle is named in the record for what it is: **Claude Fable
5, not a human** — an independent reader, blind-protocolled, overturnable.
All 120 banked variants were exported shuffled and unlabeled; the reader
named what each grid reads as with only the canonical reference card, and a
script joined the readings against the claims afterwards. Expectation could
not anchor a verdict it never saw.

**The reader rejected 31 of 120 (26%).** The disagreements sit where the
geometry predicts: arrows lose the most (claimed arrows reading as plus,
diamond, or nothing — arrows and plus share the full middle bar, so a
sloppy taper collapses the distinction), then plus→none and stripes_v→none.

**The gate — criteria registered before a single glyph was read — PASSED:**

| arm | held (kept-only) | noisy canon (control) |
|---|---:|---:|
| trained on kept only | **0.633** | 0.840 |
| trained on everything | 0.561 | 0.797 |

+0.072 at exactly **2.00x seed sd**, control improved. The concentration
diagnostic corroborates from the other side: under the historical protocol,
reader-kept held items score 0.580 against **0.406** on the rejected pile —
the §11.20–§11.27 ceiling's error mass sits precisely where the blind
reader said it would, measured before any retraining.

So the eight-section investigation ends with a confirmed mechanism instead
of another burial: the shift-Jaccard validator admits drawings a reader
refuses, those drawings were the training poison, and curation is the fix.
The boundaries stay honest: ambiguity was *part* of the ceiling (endorsed
data reaches 0.633, so §11.20–§11.24's feature and capacity limits remain
real); the validator is unchanged, because tightening it is a new mechanism
needing its own gate against the curation file as ground truth; and future
variant training trains on the curated bank
(`glyph_curation.json` beside the banks, every rejection stored with what
it read as instead, for any human to overturn).

### 11.27 Load dies too, and the variant chain closes on the data itself

The load gate crossed taught-set size {15, 30, 60, 90} from the pooled banks
with two anchor policies (the historical fixed 96 noisy copies, and copies
scaled to hold the light-load anchor:variant ratio), against a pooled held
set drawn first. Two pre-registered claims: load is supported if a smaller N
beats N=90 beyond seed sd under the fixed anchor; the anchor remedy is
supported if scaling rescues N=90 beyond seed sd.

**Neither survived.** The fixed-anchor curve is flat — 0.575/0.571/0.588/
0.588, "best" N=60 over N=90 by +0.000 at sd 0.040: six times the variant
mass, no damage on a pooled target. Anchor scaling keeps the control
healthier (0.871 vs 0.820 at N=90 — the one visible trace of anchor
thinning) but buys +0.017 at sd 0.047 on the held set. NOT SUPPORTED, twice.

Three gates now triangulate §11.25's alarming regression: styles transfer
(§11.26), volume is harmless on a pooled target (here), and the original
bank is simply the hardest sub-target (within-original 0.504 vs within-new
0.615). The -0.073 at -1.05x sd was a borderline-noise event on the hardest
slice, not a mechanism. The full suspect list for the variant ceiling is
dead: position (§11.21), scale (§11.22), parts (§11.24), style (§11.26),
load (§11.27) — capacity (§11.22) the one real, bounded lift. What remains
is the data: shift-Jaccard validation admits drawings a reader might not
call their label, and nothing trains past ambiguous targets. Measuring that
needs a human-labeled sample — the learn queue's "ask the human second"
territory, a curation mechanism rather than another training gate. The
chain rests there, characterised: this data, these features, this net sit
at 0.54–0.62 held-variant recognition, and every cheaper explanation was
measured and buried beside it.

### 11.26 One distribution after all: §11.25's diagnosis dies, and the suspect is load

§11.25 left "off-distribution interference" standing. This gate made it
falsifiable — the one-distribution hypothesis: variants are one learnable
distribution regardless of drawing voice, so a net trained on one bank should
recognise the other about as well as a bank recognises its own held half.
Four arms at hidden 64, cross arms training on **whole** banks (twice the
within arms' data — biased in favour of transfer, so rejection would have
been conservative); rejection pre-registered as both cross arms falling more
than one sd below their matching within arms:

| arm | variant acc | noisy canon (control) |
|---|---:|---:|
| within-original | 0.504 | 0.879 |
| within-new | 0.615 | 0.859 |
| cross original->new | 0.556 | 0.824 |
| cross new->original | 0.460 | 0.809 |

Gaps: -0.059 at sd 0.115 and -0.044 at sd 0.092 — both inside one sd.
**NOT REJECTED.** The voices are not meaningfully different distributions,
§11.25's parting diagnosis is measured wrong, and that is the second time
running that a section's parting diagnosis died in the next section's gate
(§11.20's "position" died in §11.21). The reports for this gate say
REJECTED / NOT REJECTED, never PASS — keeping or discarding a hypothesis is
not a capability.

What the numbers point at instead — a hypothesis, not a finding: **load**.
§11.25's enlarged arm (94 variants taught) scored 0.466 on the original held
set; this gate's cross arm trained on the 67 new variants *alone* scores
0.460 on the whole original bank — the 27 same-style variants bought almost
nothing. Every heavily-loaded arm lands in the 0.46–0.56 band; the
lightly-loaded within arms sit at the top of theirs. The suspect is the
fixed canonical anchor thinning as variant mass grows (§11.21's augmentation
collapse was this at 25x). Its gate: vary taught-set size at fixed
composition from the pooled banks, fixed pooled held set. Until it runs, no
more drawing tokens — transfer works, so volume is not what binds.

### 11.25 The data lever measured negative, and the diagnosis is distribution

§11.24 ended with "the remaining lever is data". This gate pulled it — with
the evaluation frozen first, which is the design decision everything below
depends on: held sets split only from the original 53-variant bank, new
variants training-side only, so the target could not move with the data.

**The drawing, one voice at a time, smallest first** (fresh kept / rejected):
katarau-9b 12/32 — the 9B *can* draw, unlike §11.20's 7B — cydonia-22b 15/15,
qwen3.8-27b 11/31, gemma-4-31b 6/32 (mostly duplicating its own earlier
draws: the measured drawer is mined out), qwen3.6-35b-a3b 18/22, command-r
5/14. And qwen3-48b: **0/0** — a runaway reasoner that burns any token budget
in hidden reasoning under this prompt and, unlike every model §11.13
measured, is *not* tamed by ``reasoning_effort="none"``. 67 fresh validated
variants staged.

**The gate failed in the direction nobody wants:**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| baseline (original teach half) | 0.539 | 0.867 |
| enlarged (+67 new variants) | 0.466 | 0.766 |

-0.073 at -1.05x seed sd, control down with it. More validated data made
everything worse. Three post-hoc probes — labelled post-hoc — cleared the
easy suspects: class balancing recovers the control but not the held set
(0.496); hidden 96 scores 0.504 and hidden **128** scores 0.526, so twice the
champion width still loses to no-new-data. What remains is
**off-distribution interference**: the held set is original-era drawing
style, the additions are five other voices' styles, and the mixture pulls
features away from the distribution the frozen evaluation measures.

The claim boundary matters: this says new-style data does not help
*old-style recognition* — not that the new data is worthless. The successor
question is whether "glyph variant" is one distribution at all: train on one
voice's bank and test on another's, both directions, before any more drawing
is worth the tokens. The 67 stay staged (`glyph_variants_new.json`), never
merged into the frozen bank.

### 11.24 Part-based features measured nothing, and the candidate list is spent

The last of §11.21's three mechanisms got its gate: `part_features` reads the
sixteen overlapping 2x2 patches as 4-bit patterns and histograms them — local
shapes counted with the positions thrown away — plus the ten row/column means;
`hybrid_features` appends the 16 bins to the 30 positional numbers. Three arms
at hidden 64 (the §11.22 champion), so the gate varies representation only:

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional/64 (reference) | 0.539 | 0.867 |
| parts/64 | 0.397 | 0.453 |
| hybrid/64 | 0.539 | 0.867 |

**FAIL, in both directions.** Parts alone collapse the control — local
texture cannot even hold noisy canonicals apart, because two flipped pixels
perturb patch counts more than pixel positions. The hybrid buys nothing:
-0.000 at seed sd 0.106. And the invariance's limit lands exactly on the hard
cases: patch counts are translation-invariant only while a shape's one-cell
halo stays in-grid, so the border-hugging variants — the difficult ones —
lose their boundary patches to clipping (both halves of that are pinned in
`tests/test_parts_gate.py`).

**The exact zero is the section's second finding.** The hybrid's 8-seed hit
total ties the reference's at 125/232, and their control totals tie too,
while every per-seed value differs — a coincidence so defect-shaped it was
debugged as one. The chase cleared the arms and caught a real defect in the
report instead: the contrast was computed "best-overall vs reference", which
degenerates to a row of exact zeros whenever the reference wins — a FAIL
whose printed evidence says nothing. The registered text said "the best
*parts* arm"; the code now matches the registration, and the honest scale of
the null (sd 0.106) is what that fix exposed.

With this, every mechanism named for the variant problem is measured:
position (§11.21, no), scale (§11.22, no), capacity (§11.22, the one real
lift), parts (here, no). Hidden 64 still fails 46% of held variants, and the
remaining lever is **data** — 26 taught variants per split is what every
mechanism starved on, and §11.20 already measured which models can draw more.

### 11.23 The adoption gate: §11.22's pass stops at the shape it was measured on

§11.22 ended by refusing to change the forge default silently, calling it "a
sizing decision with a measured disk cost". This gate is that measurement —
and it is a **FAIL** that protected the deployed system from its own passing
gate.

The trap it was built against: §11.22's monolith — one net over all eight
glyph families — is a protocol inherited from §11.20, and **the deployed forge
builds nothing of that shape**. It builds four family experts of two classes
each (lines, arrows, frames, crosses) at hidden 32. A width that under-fits an
8-way problem may be perfectly adequate for a 2-way one, so the capacity
result had to be re-asked at the deployed shape before it could touch a
deployed constant. 2x2 again: width {32 deployed, 64 candidate} x corpus
{deployed recipe, deployed recipe + banked variants}, held variants scored
within their family expert (chance 0.5 — not comparable to the monolith
gates), and the real cost forged end-to-end at both widths:

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| hidden 32 (deployed) | 0.784 | 1.000 |
| **hidden 32 + variants** | **0.797** | 0.996 |
| hidden 64 | 0.754 | 1.000 |
| hidden 64 + variants | 0.772 | 1.000 |

The registered contrast — 64+variants vs 32+variants — is **-0.026 at -1.06x
seed sd**: the candidate width is *worse* at the family shape, and the library
it would forge costs **82,820 bytes against 42,595** (+94%) for that
regression. Capacity binds the monolith, not the family experts. The deployed
default survives — measured, not unquestioned — and `tests/` now pins both the
verdict and the `--hidden 32` default so neither drifts on the monolith's
evidence.

Honesty notes, both in the gate's docstring: the control sits at ceiling
because two well-separated prototypes genuinely are that easy — it moved off
1.000 in one arm, so it is not §11.11's structurally-unfailable defect — and
variant-teaching buys +0.013 at the deployed shape, about what it bought the
monolith in §11.20. The 53 banked variants remain data waiting on a better
mechanism; **part-based features** is the one §11.21 candidate still holding
an unmeasured claim to a gate.

### 11.22 Capacity was the binding constraint, and scale normalisation measured useless

§11.21 corrected the problem statement to "structure and scale" and named three
candidate mechanisms — scale normalisation, part-based features, capacity —
"each deserving their own gate". This gate took the two cheap ones, and took
them **factorially**, because §11.21 is precisely the demonstration of what
uncrossed arms do: a mechanism can help, hurt, or do nothing depending on what
it is combined with.

The mechanism built for it is `scale_normalize`: stretch the set-pixel bounding
box to fill the grid by nearest-neighbour sampling, with two guards reasoned
before writing — **one uniform factor** for both axes, because independent
stretching would turn a single bar into a full block and destroy exactly the
stripes-versus-square distinction; and **centred placement**, reusing
`center_glyph`'s rule, so scale and position normalise together.
`scaled_features` sits beside `features_of` and `centered_features` for the
same reason as before: deployed experts trained on positional features must
not have the definition changed under them. A pre-gate probe motivated the
second factor: hidden (64,) lifted *both* held-variants and the control in a
sweep — while depth ((32,16)) collapsed — so any representation claim had to
survive being crossed with width.

**The gate PASSED — on the arm neither prior diagnosis predicted:**

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional/24 (§11.20's arm) | 0.448 | 0.758 |
| **positional/64** | **0.539** | **0.867** |
| scaled/24 | 0.422 | 0.770 |
| scaled/64 | 0.457 | 0.867 |

Capacity alone: **+0.091 at 1.86x seed sd**, with the control not merely held
but improved. Scale normalisation does nothing: worse than the reference at
hidden 24, and at hidden 64 it *loses* to plain positional features — the
information the normalisation discards (where and how large the glyph was)
turns out to be worth more than the alignment it buys. It joins `center_glyph`
as machinery that is correct at what it does, tested, and **not adopted**,
because it measured below the alternative.

**The diagnosis chain closes on the least glamorous candidate.** §11.20 said
the features were position-bound; §11.21 built centering and measured that
wrong (62% of variants already centred); this gate measures the scale half
wrong too. The 30→24→8 ternary network was simply **under-fitting
everything** — variants and canonicals alike — and widening it lifts both.
Three gates, each correcting the previous one's diagnosis, and the answer was
never representational: it was the width of one hidden layer. This also
retro-illuminates §11.21's augmentation collapse — a net that under-fits eight
families cannot absorb eight families crossed with twenty-five positions —
though whether augmentation survives at hidden 64 was not crossed here and
remains unmeasured.

Two honest limits. Hidden 64 still fails 46% of held variants, so capacity is
the *binding* constraint on this data, not a solution — the remaining
candidate from §11.21's list, part-based features, keeps its claim to a gate.
And the deployed forge default (hidden 32) is **not changed by this**: a wider
expert is a larger shard in a library whose size ratios are load-bearing (§6),
so adopting 64 is a sizing decision with a measured disk cost, not a silent
constant bump. The gate lives at `experiments/capacity_gate.py`, named for
what won.

### 11.21 The translation-capable representation was built, and the problem moved

§11.20 diagnosed the variant-training failure as position-bound features and
banked the 53 validated variants for "the day a translation-capable
representation gets its own gate". Both were built:
`center_glyph` translates any glyph so its mass sits at the grid centre —
the O(1) form of the shift-search the variant validator already needed — and
`centered_features` runs the same 30 features over it, so a glyph and its
translation produce identical vectors. `features_of` is untouched, because
deployed experts were trained on positional features and changing the
definition under them would silently break existing libraries.

**The gate was factorial** — two representations crossed with two training
sets — because the claim is an interaction, and only the four-way table
attributes a gain. It failed:

| arm | held variants | noisy canon (control) |
|---|---:|---:|
| positional | 0.435 | 0.773 |
| positional+variants | 0.448 | 0.758 |
| centered | 0.397 | 0.785 |
| **centered+variants** | **0.405** | 0.746 |

Not merely no gain: on the **off-centre subset the mechanism was built for**,
centring is worse than nothing — 0.319 → **0.200**. Two measurements explain
it. 33 of 53 variants are already centred (models draw centred), so the
mechanism is the identity on 62% of the data. And centring the 20 off-centre
variants translates them into the most contested region of the grid, where all
eight families' canonical pixels crowd — position is normalised and the
residual scale mismatch (a shrunk plus is *small*, not just displaced) now
competes exactly where discrimination is hardest.

**The standard alternative failed harder.** Shift augmentation — every
training row also at ±2-cell offsets — collapsed the canonical control from
0.711 to **0.348**. A 30→24→8 ternary network cannot represent eight families
crossed with twenty-five positions; asked to, it stops representing the
families.

**What the double failure buys is a corrected problem statement.** The
variants' difficulty is **structure and scale, not translation**: 62% sit
exactly where their canonical sits and still only ~44% are recognised. §11.20's
diagnosis was importantly wrong, and only building its remedy revealed that.
The mechanisms that would address the real difficulty — scale normalisation,
part-based features, or plain capacity — are each new machinery deserving their
own gate. `center_glyph` stays: it is correct at what it does, tested, and the
validator-side shift matching that motivated it remains in service.

### 11.20 Drawing distillation reached the experts, and failed on their features

The routing distillation taught the router how people *phrase* things. This
pointed the same idea at the pattern experts: a local model draws validated
**variants** of each canonical glyph — a plus in a corner, offset stripes, a
thick square — and the gate asks whether training on them improves recognition
of variants never seen, without costing the canonicals.

**Which model can draw inverts which model can phrase.** Probed before
building: the 7B that carried the routing distillation echoed the canonical and
mislabeled a diagonal as a plus — zero usable of three — while the 31B drew a
shifted plus and a thick plus. Smallest-first was measured right for text and
wrong for drawing. Neither is a principle; both are task measurements.

**The validator burned through two wrong metrics, measurably.** Plain cell
agreement scored offset horizontal bars — unambiguously stripes to a reader —
**0.000** against `stripes_h` (perfectly anti-aligned) and 0.4 against `plus`,
rejecting a correct variant as mislabeled. Shift-invariant agreement fixed that
and over-corrected, because sparse glyphs agree on empty cells almost
everywhere. **Jaccard over set pixels under ±2-cell shift** is what
discriminates: offset stripes pass, a diagonal labeled "plus" is rejected as
resembling `cross`, and 53 of ~120 generated glyphs survived validation.

**The gate failed, and the diagnosis is §11.5's lesson arriving a third way.**

| | baseline | with variants |
|---|---:|---:|
| held-out variants | 0.435 | 0.448 (+0.013, **0.23x** seed sd) |
| noisy canonicals (control) | 0.773 | 0.758 (held within sd) |

Training on half the variants barely helps with the other half, because the
expert features are **position-bound**: 25 raw pixels plus five row means, so a
corner plus shares almost no feature mass with the centered plus anchoring its
family. The validator needed shift-invariant matching to judge these variants
fairly — and the experts have the identical blindness in their representation,
where a fix in the metric cannot reach. §11.5 found that a representation which
merely re-encodes existing features cannot transfer; this is its converse: data
that only a better representation could exploit, arriving before the
representation exists. The variants are validated and kept
(`vault/glyph_variants.json`) for the day a translation-capable representation
gets its own gate. That mechanism — shift augmentation or invariant features —
is new machinery, not a retry of this run.

### 11.14 A context window in memory, with disk as the reference

The working memory this system had was the wrong shape for a machine where
memory is the binding constraint. `SystematicMemory` keeps **every** episode in
a Python list and bounds "working memory" to the last 16 by *count*;
`working()` rebuilds a dict over the whole log per call. Resident cost grows
without limit, the bound is in the wrong unit, and nothing that falls out leaves
a trace — recall means scanning.

`ultraquant/memory/context.py` is the other arrangement, and it is two
structures sized very differently:

* **Resident content** — recent turns held whole, bounded in **bytes**.
* **A reference index** — for every turn ever written, 24 bytes: id, byte
  offset, and a 64-bit membership signature. Three `array('Q')` buffers, not N
  objects. This is what makes the disk addressable without being resident.

Recall screens the index with AND + `bit_count` and seeks to the byte offsets of
the winners. Measured: **24 B of index per turn**, so a million turns of history
indexes in ~24 MB while the window itself holds whatever the budget allows.

**The gate passed, and three things had to be fixed first — two of them in the
mechanism, one in the gate.**

*The signature carried about one bit.* The first version mixed a per-token CRC32
seed with the bit index, also via CRC32. CRC32 is **linear over GF(2)**, so
every bit index reduced to the same linear functional of the seed differing only
by a constant flip. Three unrelated sentences all produced
`0xaaaa5555aaaa5555`, recall returned the same three turns for four different
questions — **and the gate reported a comfortable pass at 6.52x seed variance**.

*A correct SimHash was still the wrong tool.* BLAKE2b fixed the degeneracy and
ranked *unrelated chatter above the answer*: against "how tall is the tower",
chatter scored Hamming 23 and the answer 25. SimHash is a random projection and
needs many terms for its per-bit vote sums to be stable; a conversational turn
has four to six content tokens, so each bit is the sign of a sum of five votes
and flips on noise. A **membership bitmap** — each token sets two bits, scored
by how much of the query the turn covers — is the right structure for short
text, because the question is set overlap rather than vector angle.

*The gate's seeds varied nothing.* With a fixed fact list merely shuffled, every
seed produced the identical delta, because recall depends on vocabulary overlap
and not on arrival order. The zero-variance guard added in §11.13 caught it and
failed the run. Seeds now sample a different fact subset per history.

*And its control was at ceiling.* Scored at `top_k=3` the distractor control sat
at **1.000** — a query only had to rank the answer in its top three, which
nothing plausible could fail. That is the §11.11 defect reappearing inside the
gate written to avoid it. At `top_k=1` it reads 0.979: off ceiling, and able to
fall.

| condition | result |
|---|---:|
| buried questions, forgetting baseline | 0.000 |
| buried questions, recall | **0.896** (+0.896, **10.39x** seed sd) |
| distractors, strict `top_k=1` (**control**) | 0.979 |
| answer still resident (**control**) | 1.000 |
| peak resident against a 512 B budget | **512 B** |
| index over 66 turns | 1,584 B — **24 B/turn** |

**What it is not.** Not a KV cache. A KV cache holds computed attention state
and discards it on overflow — information genuinely lost. This holds text and an
index, so an overflowed turn is *recoverable*: better in that nothing is lost,
worse in that a recalled turn must be re-read and the signature can fetch the
wrong one. 0.896, not 1.000, is what that costs.

| stage | verdict |
|---|---|
| 0 — modality-agnostic library | **PASSED** |
| 1 — shared learned encoder | **FAILED**, twice, honestly (§11.5) — reordered the roadmap |
| 2 — blackboard composition | **PASSED** (§11.6) |
| 3 — self-proposed concepts | **PASSED** (§11.8) |
| 4 — goal-directed action | **PASSED** (§11.9) |
| 5 — grounded language | **PASSED** (§11.10) |
| hypervectors (unstaged) | half a pass (§11.7) — structure yes, retrieval no |
| external encoder (unstaged) | **FAILED** (§11.11) — on its own rebuilt control |
| LLMLS teacher panel | built, not a capability gate (§11.12) |
| router abstention | **PASSED** (§11.15) — 0.025 -> 0.975 at 12.57x sd, control held |
| plural folding | **PASSED** (§11.15) — 0.500 -> 1.000, abstention unchanged |
| self-reinforcement | **measured** (§11.16) — entrenches errors 3.9x; repeat training degraded routing 0.875 -> 0.750 |
| confirmation-gated reinforcement | **PASSED** (§11.17) — margin 2.33 -> 0.60 at 7.75x sd, transfer held |
| glyph-variant distillation | **FAILED** (§11.20) — +0.013 at 0.23x sd; data kept |
| translation representation | **FAILED twice** (§11.21) — centering hurt its own target, augmentation collapsed the control; the real difficulty is structure and scale |
| capacity vs scale (variants) | **PASSED** (§11.22) — hidden 64 lifts held variants +0.091 at 1.86x sd and improves the control; scale normalisation measured useless; §11.21's "structure and scale" resolves to under-fitting |
| width adoption at the deployed shape | **FAILED** (§11.23) — hidden 64 is *worse* for the 2-class family experts (-0.026 at -1.06x sd) at +94% forged bytes; the deployed default survives measured |
| part-based features | **FAILED** (§11.24) — parts alone collapse the control to 0.453; the hybrid buys -0.000 at sd 0.106; §11.21's candidate list is spent and the remaining lever is data |
| the data lever | **FAILED** (§11.25) — 67 new validated variants from six voices score -0.073 at -1.05x sd on the frozen held set; balance and capacity cleared post-hoc; the diagnosis is off-distribution interference between drawing styles |
| one-distribution hypothesis | **NOT REJECTED** (§11.26) — cross-bank transfer holds within one sd both directions; §11.25's style diagnosis dies; the surviving suspect is variant load against a thinning canonical anchor |
| load and its anchor remedy | **NOT SUPPORTED, twice** (§11.27) — the size curve is flat (+0.000 at sd 0.040 across a 6x mass range); §11.25's regression reclassified as borderline noise on the hardest slice; the chain closes on data ambiguity, measurable only with human labels |
| blind curation | **PASSED** (§11.28) — a blind independent reader (Claude Fable 5, named as such) rejected 26% of the banks; dropping those lifts endorsed-held recognition +0.072 at 2.00x sd, and the old ceiling's errors concentrate in the rejected pile (0.406 vs 0.580) |
| spreading-activation inference | **PASSED** (§11.30) — self-initiated derivation answers 0.938 of chain/combination questions at 8.10x sd with zero decoy falls and recall untouched; convergence, not completion, and the trail is the answer |
| consolidation + truth maintenance | **PASSED narrowly** (§11.31) — a confirmed derivation extends reach past the activation floor (+0.625 at 1.21x sd) with zero stale answers and zero repetition leaks; the flush-resurrection defect found and fixed on the way |
| validator tightening | **FAILED** (§11.32) — margin-and-floor recalls 0.050 of the reader's rejections at an unstable operating point; the reader judges shape gestalt, blind reading becomes the permanent admission protocol, and the built-but-not-adopted shelf gains strict_contradiction |
| curiosity from failed inference | **PASSED** (§11.33) — a refused convergence asks for its missing premise through the learn queue; the fail->ask->learn->infer loop closes in 0.750 of worlds at 1.62x sd with 1.000 premise precision, zero spam, zero ungrounded asks |
| the ladder | **PASSED narrowly** (§11.34) — hint-guided climbing closes depth-4/5 questions at +0.625 (1.21x sd) using exactly the depth-minimal confirmations (1.80 vs 1.80); bottom revision retracts the whole ladder, zero stale |
| compound questions | **PASSED** (§11.35) — conjunctions decompose through the full single-question machinery: +0.778 full answers at 1.76x sd, missing halves named at 1.000, arithmetic decoys held, zero malformed curiosity keys; adjective brittleness recorded as the next claim |
| adjective tolerance | **PASSED** (§11.36) — one unknown token reads as a modifier on the inference path only (+0.667 at 1.35x sd), named in every answer, zero unknown-entity fabrications in either arm; the bridge requirement is the safety proof |
| the paraphrase gate | **PASSED** (§11.37) — reorder and verbose forms hold at 1.000 with zero fabrications; the synonym wall measured at 0.000-asserted (right value still delivered hedged); embeddings buy exactly the direct half at cosine>0.75 with zero falls, and fabricate 0.833 at 0.65 |
| the density gate | **PASSED** (§11.38) — chains 1.000, fabrication 0.000, recall 1.000 from 100 to 10,000 colliding-token facts, after four failures became architecture: subject consistency, origin coverage, subject-prefix addressed buckets, order as evidence |
| the live suggester | **PASSED** (§11.39) — the synonym wall moves 0.000 -> 0.833 on the live path at 2.04x sd with zero decoy falls in either arm; the positional rule killed the attribute-anchor fabrication; the chained ceiling stays the spread's job |
| panel training run | **run, and honest** (§11.40) — three panel defects fixed (missing reasoning remedy, template-token positions, list-marker positions); the re-run's zero corroborations is exact-match integrity, ~50 claims quarantined awaiting independent evidence; containment corroboration registered for its own gate |
| containment corroboration | **PASSED** (§11.41) — elaborations jointly back their shared core (+0.600 at 1.16x sd) with zero negation falls and zero paraphrase falls; the library learned its first panel facts ('language is communication', 3 voices), with one stored fact flagged for human veto |
| unit conversion | **PASSED** (§11.42) — definition-table conversion answers 0.643 of cross-unit combinations at 1.29x sd with zero refusal falls; cross-family, unknown-unit, and bare-number combinations refuse exactly as §11.30 required |
| self-study | **PASSED** (§11.43) — the loop closes 0.667 of its own seeded gaps at 1.29x sd with ZERO invented-term corroborations across two live passes and zero wrong subjects; contested concepts stay open by design; ':study' ships |
| plasticity | **PASSED** (§11.44) — reinforcement breaks retrieval ties (+0.571 at 1.11x sd) with zero entrenchment falls and better token matches always winning; rule 3 finally reaches the fact layer |
| the VRAM tier | **PASSED** (§11.45) — resident layers at 3.6e-15 parity, 0.061 -> 0.028 ms per query at 3.26x rep sd on an RTX 4090, accounting balanced; the disk/DRAM/VRAM split is measured at all three tiers |
| storage split + sequential training | **live** (§11.19) — context window wired, one-voice-at-a-time training; run 4 leaked a template token, was caught by the decoy gate, and rolled back |
| route correction (`:correct`) | **works** (§11.18) — deployed routing 0.750 -> 1.000; withdrew the claim that `:learn` could do it |
| context window + reference index | **PASSED** (§11.14) — +0.896 at 10.39x sd; two wrong signatures and a ceilinged control fixed first |
| offline distillation | **PASSED narrowly** (§11.13) — +0.062 at 1.79x sd; first mechanism failed, it does not scale, and the margin was inflated until corrected |

Every gate was stated before its experiment ran; two experiments had their own
flaws caught and recorded (§11.5's ceiling, §11.7's oracle-baseline); one
passing stage had a feature deleted for measuring nothing (§11.6's escalation
heuristic). What the system now is: a pattern library that pages itself, reads
compound scenes by composition, proposes its own concepts, plans multi-step
actions from goals, and says what it sees in a language it induced — every
capability behind a measured gate. What it still is not is what §11.4 said at
the start: open-ended goal formation, transfer across structural novelty, and
problem reformulation are not deferred work. They are unsolved.
