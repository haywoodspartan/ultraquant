"""Ask about members of the kinds the library's subjects belong to (§11.170).

The dictionary lists what exists: every instance of "African country", every
leaf under "chemical element". The subjects holding an attribute name their
kinds, and the members the library does not hold yet become questions in the
form the library already asks that attribute in.

Words have senses, so each step is guarded against reading the wrong one:

* **Coherence.** Half the subjects must share an ancestor among their nouns.
* **Core kinds.** Subjects with one noun sense name the kinds. A subject with
  several ("Quebec", a province and a city; a novel's title, also its hero)
  counts only its first sense of a kind already named.
* **Aliases.** A leaf is asked about only when the dictionary spells the
  attribute's value into its members ("atomic number 30" for zinc), and never
  when a held subject already has that value: "ununbium" is copernicium.
"""

from collections import Counter, defaultdict, deque

from . import seeds, targets
from ultraquant.lexicon import normalize_word
from ultraquant.memory.factshards import normalize_subject


def noun_senses(lexicon, subject) -> list[str]:
    """The synset ids of the subject's noun senses, in the dictionary's order."""
    return [sense["synset"] for sense in lexicon.senses(subject)
            if sense["pos"] == "n"]


def direct_kinds(lexicon, synset_id) -> list[str]:
    """What a synset is an instance of, or else what it is a kind of."""
    record = lexicon.synset(synset_id) or {}
    return list(record.get("instance_hypernym") or record.get("hypernym") or [])


def _ancestors(lexicon, synset_id) -> list[str]:
    """Breadth-first through instance links, then kind links, never itself."""
    seen, found = {synset_id}, []
    queue = deque([synset_id])
    while queue:
        record = lexicon.synset(queue.popleft()) or {}
        for parent in (*record.get("instance_hypernym", ()),
                       *record.get("hypernym", ())):
            if parent not in seen:
                seen.add(parent)
                found.append(parent)
                queue.append(parent)
    return found


def coherent(lexicon, subjects) -> bool:
    """Whether one ancestor of their noun senses is reached by half the subjects."""
    # Lexicon.kinds climbs every part of speech; only nouns name kinds here.
    subjects = list(subjects)
    reached = defaultdict(set)
    for subject in subjects:
        for sense in seeds.noun_senses(lexicon, subject):
            for ancestor in _ancestors(lexicon, sense):
                reached[ancestor].add(subject)
    if not subjects or not reached:
        return False
    return 2 * max(map(len, reached.values())) >= len(subjects)


def core_kinds(lexicon, subjects) -> dict[str, int]:
    """Direct kinds of two or more subjects, read through unambiguous nouns."""
    counts = Counter()
    ambiguous = []
    for subject in subjects:
        senses = seeds.noun_senses(lexicon, subject)
        if len(senses) == 1:
            counts.update(seeds.direct_kinds(lexicon, senses[0]))
        elif senses:
            ambiguous.append(senses)
    # Only the unambiguous name the core, so "Quebec" counts as a province
    # beside Ontario and never makes cities a kind of its own.
    core = set(counts)
    for senses in ambiguous:
        for sense in senses:
            kinds = [kind for kind in seeds.direct_kinds(lexicon, sense)
                     if kind in core]
            if kinds:
                counts.update(kinds)
                break
    return {kind: count for kind, count in counts.items() if count >= 2}


def is_alias(value, held_values) -> bool:
    """Whether a leaf's value is one a held subject already has."""
    return normalize_word(value) in held_values


def _members(lexicon, synset_id) -> list[str]:
    return (lexicon.synset(synset_id) or {}).get("members") or []


def _is_held(members, held_same) -> bool:
    return any(normalize_subject(member) in held_same for member in members)


def _witnessed(lexicon, attribute, values) -> bool:
    """Whether half the held subjects have their value spelled as a member."""
    count = 0
    for subject, value in values.items():
        spelled = normalize_word(f"{attribute} {value}")
        count += any(normalize_word(member) == spelled
                     for sense in seeds.noun_senses(lexicon, subject)
                     for member in _members(lexicon, sense))
    return 2 * count >= len(values)


def dictionary_targets(memory, stash, lexicon) -> list[targets.Target]:
    """Ask about the unheld members of the kinds each attribute's subjects share."""
    facts = defaultdict(dict)
    for key in memory.fact_keys():
        record = memory.recall_fact(key) or {}
        if record.get("subject") and record.get("attribute"):
            facts[record["attribute"]][record["subject"]] = str(record.get("value"))
    found = {}
    for attribute in sorted(facts):
        values = facts[attribute]
        held = set(values)
        held_same = {normalize_subject(subject) for subject in held}
        held_values = {normalize_word(value) for value in values.values()}
        if not seeds.coherent(lexicon, held):
            continue
        form = targets.question_form(stash, attribute)
        if form is None:
            continue
        category, question_format = form
        kinds = seeds.core_kinds(lexicon, held)
        witnessed = _witnessed(lexicon, attribute, values)
        prefix = normalize_word(attribute) + " "
        names = []
        for kind in sorted(kinds):
            record = lexicon.synset(kind) or {}
            for synset_id in record.get("instance", ()):
                members = _members(lexicon, synset_id)
                if members and not _is_held(members, held_same):
                    names.append(members[0])
            if not witnessed:
                continue
            # A subkind is a member only where the dictionary gives its value
            # under the attribute's own name, as it does for the held subjects.
            for synset_id in record.get("hyponym", ()):
                members = _members(lexicon, synset_id)
                if not members or _is_held(members, held_same):
                    continue
                spelled = [word[len(prefix):] for word in map(normalize_word, members)
                           if word.startswith(prefix)]
                if not spelled or any(seeds.is_alias(value, held_values)
                                      for value in spelled):
                    continue
                names.append(members[0])
        for name in names:
            found[attribute, name] = targets.Target(
                category, name, question_format.format(subject=name), attribute)
    return [found[key] for key in sorted(found)]
