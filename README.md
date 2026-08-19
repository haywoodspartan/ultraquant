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
| **From-scratch quantum engine** | exact statevector sim, Mottonen encoding, parameter-shift gradients, kernels, ZNE, Grover; IBM QPU / BlueQubit optional |
| **Content routing** | sketch-screened cosine over stored prototypes: 47x faster than exact scan at 20k categories, exact rerank always |
| **Composition** (blackboard) | reads compound scenes a monolithic classifier scores 0.000 on (0.539, at 2.4x seed variance) |
| **Self-proposed concepts** | clusters what it could not place and asks *you* what to call the group (ARI 1.000, class count never supplied) |
| **Goal planning** | multi-step action sequences from goals alone — no per-task scripts; capabilities added later are usable without planner changes |
| **Grounded language** | induced lexicon, order and frame — no templates, asserted over the AST; speaks what it perceives, in three languages, at nesting depths never seen in training |
| **Self-learning loop** | finds its own gaps, asks the web first (through a found-is-not-believed quarantine), asks the human second |
| **Spreading-activation inference** | answers questions the library holds only implicitly: activation converges across stored facts (0.938 of chain/arithmetic questions at 8.10x sd, zero fabrication decoys fell); premises named, marked inferred, confidence diluted |
| **Consolidation + truth maintenance** | say "yes" to a derivation and it becomes recallable memory usable as a premise (+0.625 reach at 1.21x sd); revise a premise and every conclusion resting on it retracts, recursively |
| **Curiosity from failed inference** | a question it cannot connect becomes a question it ASKS: the missing premise enters the learn queue at top rank, and answering it closes the fail->ask->learn->infer loop (0.750 closure at 1.62x sd, 1.000 premise precision, zero spam) |
| **Evidence accumulation** | corroboration and contradiction typed and graded from live-web measurement; belief rises by degree, conflicts become questions |
| **Learned dispatch** | cores, threads and tiers decided by machine-learned experience — three brains (classical net / variational quantum circuit / one-qubit-per-core committee) in a measured shootout; 13x regret removed vs the static preference walk |

Where a mechanism failed its gate, that is in the book too: the shared-encoder
stage failed twice, honestly, and reordered the roadmap; hypervector retrieval
passed on structure and failed on similarity; a rehearsal mechanism and an
escalation heuristic were deleted for measuring nothing.

## Quick start

**Windows:** double-click `UltraQuant.bat` — no packages, no environment.
**Linux/macOS:** `./ultraquant.sh` (desktop app with Tkinter, terminal UI
without).

```
python -m ultraquant.gui                   # desktop app: 10 tabs
python -m ultraquant.tui                   # the same surfaces over SSH
python -m ultraquant.interpreter.chat     # terminal chat
python -m ultraquant.forge.build --synthetic 64 --compare
python -m unittest discover -s tests      # 1327 tests, ~4 min
```

In the chat, try:

```
#####            <- paste a 5x5 glyph; it says what it sees, in its languages
:learn           <- the model surveys its own gaps and asks you questions
:learn research  <- it tries the web first (':online on'), quarantined
:learn panel ... <- or asks local models, counted by independent lineage
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
  memory/      episodic/semantic stores - facts as bucket shards
  interpreter/ thought pipeline - chat CLI - stash quarantine - learning mode
  forge/       build libraries from scratch - deployment languages - seed facts
  native/      C++/CUDA accelerators - the learned dispatch scheduler
  storage/     NVMe-oF / Ceph / SAN backends - RAM tier - paged index
  experiments/ the gates: every capability's pre-registered measurement
tests/         1327 tests across 71 modules
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
