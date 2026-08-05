# UltraQuant — Shard Library & Chat/Interpreter Specification (v0.1)

Extends SPEC.md (conventions binding: pure stdlib, seeded `random.Random` only,
type hints + docstrings, Windows-safe paths). This layer realizes the project's
guiding idea: **the model is a library of patterns of learned data, catalogued
associatively like a human brain**. Weights and knowledge live as *shards*
(layers/slices). Shards pack into large `.uql` library files with an offset
index; they are read back as byte-range chunks **on demand** and swapped in/out
of RAM under an explicit budget — never load the whole store. Access reinforces
associations (like recall strengthening a memory trace); consolidation packs
hot shards into libraries (like sleep consolidation).

New layout (additions only; `__init__.py` files already exist — do not modify):

```
ultraquant/shards/
  vault.py         ShardVault + .uql container format
  budget.py        ShardCache (LRU under a byte budget)
  router.py        CategoryRouter (associative routing)
  scale_demo.py    on-demand swap demonstration at scale   (written by integrator)
ultraquant/experts/
  moe.py           ExpertPool: per-category ternary nets as shards
ultraquant/interpreter/
  codefunc.py      SafeCodeRunner (sandboxed code function)
  webaccess.py     WebAccess (gated internet)
  stash.py         ContemporaryStash (web quarantine: analyze before FACT)
  thoughts.py      the predefined thought pipeline          (integrator)
  selflearn.py     SelfLearner + consolidation              (integrator)
  chat.py          REPL / script / one-shot CLI             (integrator)
tests/
  test_shards.py  test_experts.py  test_codeweb.py  test_interpreter.py
```

Run ONLY named test modules (e.g. `python -m unittest tests.test_shards -v`) —
NEVER `discover` (other workstreams are running concurrently). Do not touch
README.md, ultraquant/native/**, ultraquant/pattern/**, ultraquant/bench.py.

---

## 1. `ultraquant/shards/vault.py`

- `class ShardIntegrityError(RuntimeError)`
- Shard catalog entry (plain dict, JSON-safe):
  `{shard_id, category, kind, codec ("zlib"|"raw"), nbytes (stored size),
    sha256 (of stored bytes), created (UTC ISO), location ("loose"|"library"),
    library_path (str|None), offset (int), length (int),
    access_count (int), last_access (UTC ISO|None),
    associations (dict[str, float])}`  — associations are the brain-like
  keyword→strength weights, reinforced on use.
- `class ShardVault:`
  - `__init__(self, root)` — creates `root/`, `root/loose/`; catalog at
    `root/catalog.json` (auto-load if present, auto-save on every mutation).
  - `add_shard(self, shard_id, category, payload: dict, kind="expert-net",
     associations=None) -> dict` — serialize payload
    (`json.dumps(sort_keys=True, separators=(",", ":"))`, utf-8), zlib-compress,
    write `root/loose/{shard_id}.uqs`, record catalog entry (location "loose").
    Re-adding an existing id overwrites bytes and refreshes the entry but
    PRESERVES access_count/associations (learning survives retraining).
  - `load_bytes(self, shard_id) -> bytes` — loose: read the file; library:
    open the `.uql`, `seek(offset)`, `read(length)` — **only that chunk**.
  - `get(self, shard_id) -> dict` — load_bytes, verify sha256
    (mismatch → `ShardIntegrityError`), decompress, parse; bumps `touch()`.
  - `touch(self, shard_id)` — access_count += 1, last_access = now.
  - `reinforce(self, shard_id, keywords: list[str], delta: float = 0.1)` —
    `associations[k] = min(5.0, associations.get(k, 0) + delta)`.
  - `pack(self, library_path, shard_ids=None, prune_loose=False) -> int` —
    build a `.uql` container: magic `b"UQL1"` + 8-byte big-endian index length
    + JSON index (utf-8: list of {shard_id, offset, length, sha256, codec,
    category, kind}) + concatenated stored-bytes payloads. Offsets are absolute
    file offsets. Update catalog entries to location "library". If prune_loose,
    delete the loose files ONLY after re-reading each chunk from the library
    and verifying its sha256. Returns count packed.
  - `attach(self, library_path) -> int` — read ONLY magic+index (never the
    payloads), merge entries into the catalog (new ids get fresh
    access_count 0 / empty associations; known ids keep their stats). Returns
    count attached.
  - `catalog(self) -> list[dict]` (copies), `stats(self) -> dict`
    (`{shards, loose_bytes, library_bytes, total_bytes, libraries: [paths]}`).

## 2. `ultraquant/shards/budget.py`

- `class ShardCache:` — resident-set manager; the "load chunks on demand,
  swap in and out on the fly" mechanism.
  - `__init__(self, max_bytes: int)`.
  - `get(self, shard_id, loader: Callable[[], tuple[dict, int]]) -> dict` —
    hit: move to most-recent, return. Miss: call loader → (payload, nbytes),
    insert, then evict least-recently-used entries until
    `current_bytes <= max_bytes`. A single item larger than the budget is
    allowed to be resident alone (evict everything else).
  - `invalidate(self, shard_id)`, `set_budget(self, max_bytes)` (evicts down
    immediately), `resident(self) -> list[str]` (LRU→MRU order).
  - `stats(self) -> dict` — `{hits, misses, evictions, current_bytes,
    peak_bytes, budget_bytes, resident}`.

## 3. `ultraquant/shards/router.py`

- `class CategoryRouter:` — associative recall: text → ranked categories.
  - `__init__(self, vault: ShardVault, memory=None, path=None)` — optional
    persistence file (`router.json`, auto-load if exists).
  - `register(self, category, keywords: list[str])` — base keyword set.
  - `route(self, text, top_k=3) -> list[tuple[str, float]]` — tokenize
    (lowercase, alnum words), score each registered category:
    base-keyword overlap + learned keyword weights + sum of matching
    `associations` across that category's vault shards. Deterministic
    tie-break (alphabetical). Empty/no-match → [].
  - `learn(self, text, category, delta=0.1)` — strengthen tokens→category
    (internal weights) and `vault.reinforce` the category's shards.
  - `save() / load()`; `state() -> dict` for snapshots.

## 4. `ultraquant/experts/moe.py`

- `class ExpertPool:` — per-category experts, loaded through the cache on
  demand (this is the parameter paging: N experts on disk, few resident).
  - `__init__(self, vault, cache, input_dim=30, hidden=(32,), seed=0)`.
  - Shard id convention: `expert:{category}`; payload
    `{"labels": [...], "net": UltraQuantNet.state_dict(), "trained_examples": int}`.
  - `ensure_expert(self, category, labels: list[str])` — create+add a fresh
    seeded net shard if absent.
  - `predict(self, category, features: list[float]) -> tuple[str, float]` —
    `cache.get(...)` with a vault loader, rebuild the net
    (`load_state_dict`), return (label_name, confidence).
  - `train_expert(self, category, xs, ys, labels, epochs=30, lr=0.05) -> dict`
    — train (fresh or continued from stored weights), `add_shard` back
    (persisting), `cache.invalidate`, return `{"loss": ..., "accuracy": ...}`.
  - Loader nbytes = the shard's stored `nbytes` from the catalog (that is the
    budget currency; document it).

## 5. `ultraquant/interpreter/codefunc.py` — the code function

- `class CodeError(Exception)`
- `class SafeCodeRunner:` — sandboxed mini-Python via AST whitelist. This must
  be SAFE: the review stage will actively attempt escapes.
  - `__init__(self, max_ops: int = 200_000, timeout_s: float = 2.0)`.
  - `run(self, source: str) -> dict` — `{"result": <last-expression value or
    None>, "stdout": str, "defined": list[str]}`; raises `CodeError` on any
    violation, runtime error, op-budget breach, or timeout.
  - ALLOWED nodes: Module, Expr, Assign, AugAssign, FunctionDef (plain args
    only), Return, Lambda, If, For, While, Break, Continue, Pass, BoolOp,
    BinOp, UnaryOp, Compare, IfExp, Call, Name, Constant, List, Tuple, Dict,
    Set, Subscript, Slice, ListComp/SetComp/DictComp/GeneratorExp +
    comprehension, JoinedStr, FormattedValue.
  - FORBIDDEN (non-exhaustive — whitelist, don't blacklist): Import/ImportFrom,
    Attribute (sole exception: `math.<name>` where the attr is a whitelisted
    math function, Load context only), any identifier starting with `_`,
    With, Try, Raise, Class, Global, Nonlocal, Delete, Yield, Await, Starred,
    keyword arguments named with leading `_`.
  - Namespace: exactly `abs, min, max, sum, len, range, round, pow, sorted,
    reversed, enumerate, zip, map, filter, print (captured to the returned
    stdout), int, float, str, bool, list, dict, set, tuple` plus `math`
    (whitelisted attrs: sqrt, sin, cos, tan, pi, e, log, log2, log10, exp,
    floor, ceil, fabs, tanh, atan, atan2, hypot, gcd, comb, perm). Build the
    exec globals with `{"__builtins__": {}}` plus the whitelist.
  - Op budget: `sys.settrace` line-event counter → `CodeError` past `max_ops`.
    Timeout: run in a worker thread, `join(timeout_s)`; on timeout raise
    `CodeError` (document that the worker may linger — acceptable).

## 6. `ultraquant/interpreter/webaccess.py` — gated internet

- `class WebDisabled(RuntimeError)`
- `class WebAccess:`
  - `__init__(self, online: bool = False, timeout: float = 10.0,
     max_bytes: int = 262_144, user_agent: str = "UltraQuant/0.1")`.
  - `set_online(self, flag: bool)`; `self.online` readable.
  - `fetch(self, url) -> dict` — `{"url", "status", "title", "text"}`.
    Offline → raise `WebDisabled`. Scheme must be http/https (else
    ValueError). `urllib.request` with the timeout and UA header; read at
    most `max_bytes` bytes. HTML → text via an `html.parser.HTMLParser`
    subclass that drops script/style, captures <title>, collapses whitespace.
    Non-HTML content types: decode best-effort as text.
  - The interpreter only ever fetches URLs the local user typed or supplied
    via `:fetch` — no autonomous crawling. Tests use a local
    `http.server` on 127.0.0.1 (never the real internet).
  - Fetched content NEVER becomes a memory fact directly — it goes to the
    ContemporaryStash (§6b) for analysis first. WebAccess has no
    write-to-memory methods.

## 6b. `ultraquant/interpreter/stash.py` — the contemporary stash

Quarantine + analysis between the web and stored knowledge: nothing fetched
becomes FACT/Truth until it has been classified and either corroborated or
explicitly promoted; opinion and falsehood-suspects stay out of memory.

- `class StashError(RuntimeError)`
- Entry (JSON-safe dict): `{id (int), url, netloc, fetched (UTC ISO),
  claim (one sentence), classification ("unclassified"|"factual-claim"|
  "opinion"|"hedged"), status ("staged"|"corroborated"|"disputed"|
  "promoted"|"rejected"), sources (list[str] netlocs that asserted this
  claim), notes (str)}`.
- `class ContemporaryStash:`
  - `__init__(self, path)` — `stash.json`, auto-load if present, auto-save
    on mutation.
  - `add_page(self, url, title, text, max_claims: int = 8) -> list[int]` —
    split text into sentences (., !, ? boundaries; 20–300 chars); keep the
    first `max_claims`; each becomes a staged entry (dedup: an identical
    normalized claim from a NEW netloc appends to that entry's `sources`
    instead of creating a duplicate). Returns entry ids touched.
  - `analyze(self, memory=None) -> dict` — classify every unclassified
    entry: "opinion" if it contains first-person/judgment markers (i think,
    i believe, in my view, best, worst, should, amazing, terrible,
    beautiful, awful, favorite, ...); "hedged" if reportedly, allegedly,
    rumored, may, might, possibly, some say; else "factual-claim" when it
    matches a declarative shape ("<subj> is/are/was/were/has/have <rest>"
    or contains a number+unit), otherwise stays "unclassified".
    Status transitions: factual-claim with `len(sources) >= 2` →
    "corroborated"; factual-claim whose "<A> is <B>" contradicts an
    existing memory fact for key A (different stored value) → "disputed".
    Returns counts per classification/status.
  - `promote(self, entry_id, memory, force: bool = False) -> str` — only
    "factual-claim" entries with status "staged"/"corroborated" promote
    freely; "opinion"/"hedged"/"disputed" raise `StashError` unless
    `force=True`. Promotion: `memory.remember_fact(key, value,
    confidence= 0.8 if corroborated else 0.55)` where key/value come from
    the "<A> is <B>" split (fallback key: `web:{netloc}:{id}`), plus a
    `"promotion"` episode tagged ["web", "stash"] recording url + sources.
    Sets status "promoted"; returns the fact key.
  - `reject(self, entry_id, reason: str = "")` — status "rejected", note.
  - `entries(self, status=None) -> list[dict]`, `get(self, entry_id)`,
    `stats(self) -> dict` (counts by classification and status).

## 7. `ultraquant/interpreter/thoughts.py` — the predefined set of thought

- `@dataclass class Session:` — wires everything:
  `memory, vault, cache, router, experts, web, stash, coder,
   recognizer (optional), archive (optional), rng (random.Random)`.
  Factory `def build_session(root, budget_bytes=1_048_576, online=False,
  seed=0) -> Session` — constructs the full stack rooted at `root/`
  (vault at `root/vault`, memory at `root/memory.json`, archive at
  `root/artchive`, router state at `root/vault/router.json`, stash at
  `root/stash.json`).
- `@dataclass class ThoughtContext:` `text: str; session: Session;
  trace: list[dict]; response_parts: list[str]; data: dict`.
- Thought = class with `name: str` and `run(self, ctx) -> None`; each appends
  `{"thought": name, "summary": <one line>}` (+ extra keys) to `ctx.trace`.
- `PIPELINE` (module constant, exactly this order — the predefined set):
  1. `Perceive` — tokenize; classify intent ∈ {fact_statement, question,
     code, url, glyph, teach, chat} (simple deterministic rules: leading
     "calc:"/"code:" → code; contains http(s):// → url; "X is Y" /
     "remember ..." → fact_statement; 5 rows of 5 [#.] chars → glyph;
     trailing "?" → question; else chat).
  2. `Recall` — memory facts matching tokens, recent episodes, nearest
     signature if glyph-like; stash in ctx.data.
  3. `Route` — `router.route(text)`; note ranked categories and which of
     their expert shards are currently resident vs would need loading
     (via cache.resident()).
  4. `Reason` — dispatch on intent: code → coder.run; url → web.fetch,
     then `stash.add_page` + `stash.analyze(memory)` — REPORT what was
     staged (counts by classification) and explicitly that nothing became
     fact yet; offline → say so; glyph → experts.predict on the routed
     category (features = the 25 pixels + 5 row means, matching SPEC.md
     §10 classical features) or recognizer if provided; question → answer
     from best recalled fact (facts promoted from the stash count like any
     other); fact_statement → acknowledge; chat → template reply naming
     top category + a recalled fact if any.
  5. `Respond` — assemble `response_parts` into the final string
     (always non-empty).
  6. `Learn` — store the episode; fact_statement (typed by the USER, not
     from the web) → `memory.remember_fact`; auto-promote ONLY stash
     entries that reached status "corroborated" this turn (≥2 independent
     netlocs); everything else in the stash awaits manual :promote/:reject;
     reinforce router+vault associations for the categories used;
     glyph → store signature under predicted label.
- `def run_pipeline(text: str, session: Session) -> tuple[str, list[dict]]`.
  Honesty rule: replies are template-based symbolic output — never pretend
  to be a general LLM; unknown → say so and suggest `:help`.

## 8. `ultraquant/interpreter/selflearn.py` — build off what it has

- `class SelfLearner:`
  - `__init__(self, session)`.
  - `extract_facts(text) -> list[tuple[str, str]]` — "<A> is <B>" (A ≤ 4
    words), "remember[,:] K = V" / "remember that A is B" patterns.
  - `teach_glyph(self, category, labels, rows: list[str], label: str,
     epochs=15) -> dict` — noisy-augment the taught glyph (seeded flips),
    `experts.train_expert` continuing from stored weights, reinforce router.
  - `consolidate(self) -> dict` — pack hot loose shards
    (access_count >= 2) into `root/vault/library/uq_lib_{n:04d}.uql`
    (prune_loose=True), save router + memory, commit an ArTchive snapshot
    (payload: catalog summary, router state, memory stats, cache stats) —
    returns `{"packed": n, "library": path|None, "snapshot": T-id}`.

## 9. `ultraquant/interpreter/chat.py` — the Chat/Interpreter CLI

- `python -m ultraquant.interpreter.chat [--root DIR] [--online]
  [--budget-kb N (default 1024)] [--seed N] [--script FILE] [--once TEXT]`
- Free text → `run_pipeline`. Commands:
  `:help` (list everything), `:trace` (last pipeline trace, one line per
  thought), `:mem` (memory.stats), `:facts [substr]`, `:shards`
  (catalog: id, category, location, nbytes, access_count),
  `:resident` (cache.stats incl. resident list), `:budget <kb>`,
  `:pack` / `:consolidate` (SelfLearner.consolidate), `:attach <file>`,
  `:online on|off`, `:fetch <url>`, `:stash [id]` (list staged entries /
  detail), `:analyze` (stash.analyze), `:promote <id> [force]`,
  `:reject <id> [reason]`, `:code <source>` (single line),
  `:teach <category> <label>` then 5 glyph rows (in script mode the rows
  follow on the next 5 lines), `:recognize` then 5 glyph rows,
  `:snapshot`, `:quit`.
- `--script FILE`: read lines as if typed; echo `> input` then the response;
  exit 0 at EOF. `--once TEXT`: one pipeline pass, print response, exit 0.
  These make the REPL fully testable non-interactively.
- Seed all components from `--seed`. Default root: `./uq_home`.

## 10. `ultraquant/shards/scale_demo.py` — the on-demand story at scale

`python -m ultraquant.shards.scale_demo [--shards 64] [--budget-kb 96]
[--accesses 300] [--seed 0]`: synthesize N shards across ~8 categories
(payloads sized so ALL of them together far exceed the budget), attach-style
catalog, then run seeded category-skewed accesses through a ShardCache.
Print: total store bytes vs budget bytes, hit/miss/eviction counts, peak
resident bytes (must be <= budget), final resident set, and a closing line
explaining this is exactly how a 300B-parameter store would page: the catalog
picks the shard, the budget decides what stays resident. Exit 0; assert the
budget was never exceeded (peak_bytes <= budget).

## 11. Tests (unittest; named modules only; fast; no real network)

- `test_shards.py` — add/get round trip + sha verify; tamper loose file →
  ShardIntegrityError; pack → attach in a FRESH vault → chunked get equality
  with the original payloads; prune_loose leaves library-only reads working;
  re-add preserves associations/access_count; LRU: with budget for 2 of 3
  equal-size shards, access A,B,C → A evicted, then A reload evicts B (order
  + eviction count + current_bytes asserted); oversize-single-shard case;
  set_budget shrink evicts immediately; stats correctness.
- `test_experts.py` — ensure/predict on-demand load (cache miss then hit);
  train_expert persists (fresh pool + fresh cache reloads trained weights and
  keeps accuracy); two experts under a one-expert budget swap correctly;
  router: register + route ranks the right category; learn() flips a tie.
- `test_codeweb.py` — SafeCodeRunner: arithmetic result, def+call, loop with
  accumulator, math.sqrt, f-string, comprehension, print capture; CodeError
  on: import, open, __class__ (any `_` name), attribute on non-math,
  getattr-by-name absence, while True (op budget), exec/eval absence;
  results dict shape. WebAccess: offline fetch → WebDisabled; ftp:// →
  ValueError; live fetch against a threading http.server on 127.0.0.1
  serving a small HTML page (title + script-stripped text verified).
  ContemporaryStash: add_page splits claims + dedups same claim from a new
  netloc into `sources`; analyze classifies a seeded opinion sentence as
  "opinion", a hedged one as "hedged", a declarative as "factual-claim";
  2-source claim → "corroborated"; claim contradicting an existing memory
  fact → "disputed"; promote: corroborated promotes at confidence 0.8 and
  writes the fact + "promotion" episode; opinion promote without force →
  StashError, with force succeeds; reject; persistence round trip; stats.
- `test_interpreter.py` — build_session in a temp dir; run_pipeline trace
  order == PIPELINE names; fact teach→ask round trip ("the sky color is
  blue" then "what is the sky color?" answer contains "blue"); code intent
  computes; glyph recognize returns a label and stores a signature; url
  intent against two local http.servers on different ports (distinct
  netlocs, acceptable stand-in for independent sources — note it in a
  comment): first fetch stages claims WITHOUT creating any memory fact,
  second fetch of the same claim corroborates and the Learn thought
  auto-promotes exactly that claim; chat.py --script end-to-end (script
  exercising :help, a fact, a question, :code, :shards, :stash, :analyze,
  :consolidate, :quit) exits 0 and output contains expected substrings;
  consolidate produces a .uql + a T-snapshot; --once works. Everything
  seeded/deterministic; temp dirs cleaned up.
