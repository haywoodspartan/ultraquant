"""A benchmark with known answers, and subjects that do not exist.

Distilling facts out of a language model is only worth anything if the
facts that come out are right, and a model says nothing about which of
its answers are. So the distillation exam asks questions whose answers
are known, to measure how often what passes its filter is right, and
questions about subjects that were invented for this file, to which
the only right answer is "unknown". A filter that promotes anything at
all about Veltrania has promoted a hallucination.

**The known answers were chosen to be beyond dispute.** Every fact was
left out that has ever had two answers:
- disputed or split capitals (Bolivia, the Netherlands, South Africa);
- capitals that moved or were renamed recently (Indonesia, Kazakhstan);
- capitals with competing spellings (Kyiv, Nay Pyi Taw).
Authors carry the surnames and spellings a correct answer might use,
and accents are folded away before comparison.

**The invented subjects were chosen not to exist**: element names that
no periodic table has, countries that appear in no atlas and in none of
the well-known fictional ones, and novel titles specific enough that no
real book should carry them. That last one cannot be proven. A title
that turns out to be real is a false alarm against the filter, never a
false pass.

Four categories, each paired with its invented counterpart. Items are
split into calibration and held-out halves by position, so a confidence
measured on one half can be checked on the other.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ["Item", "KNOWN", "FICTITIOUS", "normalize", "is_correct",
           "is_abstention", "split"]


@dataclass(frozen=True)
class Item:
    """One question, with the answers that count as right (none if invented)."""

    category: str
    subject: str
    question: str
    answers: tuple = ()

    @property
    def fictitious(self) -> bool:
        return not self.answers


_SYMBOLS = (
    ("hydrogen", "h"), ("helium", "he"), ("lithium", "li"),
    ("beryllium", "be"), ("boron", "b"), ("carbon", "c"),
    ("nitrogen", "n"), ("oxygen", "o"), ("fluorine", "f"), ("neon", "ne"),
    ("sodium", "na"), ("magnesium", "mg"), ("aluminium", "al"),
    ("silicon", "si"), ("phosphorus", "p"), ("sulfur", "s"),
    ("chlorine", "cl"), ("argon", "ar"), ("potassium", "k"),
    ("calcium", "ca"), ("titanium", "ti"), ("chromium", "cr"),
    ("manganese", "mn"), ("iron", "fe"), ("cobalt", "co"), ("nickel", "ni"),
    ("copper", "cu"), ("zinc", "zn"), ("gallium", "ga"),
    ("germanium", "ge"), ("arsenic", "as"), ("selenium", "se"),
    ("bromine", "br"), ("krypton", "kr"), ("strontium", "sr"),
    ("silver", "ag"), ("tin", "sn"), ("antimony", "sb"), ("iodine", "i"),
    ("xenon", "xe"), ("barium", "ba"), ("tungsten", "w"),
    ("platinum", "pt"), ("gold", "au"), ("mercury", "hg"), ("lead", "pb"),
    ("bismuth", "bi"), ("radon", "rn"), ("uranium", "u"),
    ("plutonium", "pu"),
)

_NUMBERS = (
    ("hydrogen", "1"), ("helium", "2"), ("lithium", "3"), ("carbon", "6"),
    ("nitrogen", "7"), ("oxygen", "8"), ("fluorine", "9"), ("neon", "10"),
    ("sodium", "11"), ("magnesium", "12"), ("aluminium", "13"),
    ("silicon", "14"), ("sulfur", "16"), ("chlorine", "17"),
    ("argon", "18"), ("potassium", "19"), ("calcium", "20"),
    ("iron", "26"), ("copper", "29"), ("zinc", "30"), ("silver", "47"),
    ("tin", "50"), ("iodine", "53"), ("platinum", "78"), ("gold", "79"),
    ("mercury", "80"), ("lead", "82"), ("radon", "86"), ("uranium", "92"),
    ("plutonium", "94"),
)

_CAPITALS = (
    ("France", ("paris",)), ("Germany", ("berlin",)), ("Italy", ("rome",)),
    ("Spain", ("madrid",)), ("Portugal", ("lisbon",)),
    ("Japan", ("tokyo",)), ("China", ("beijing",)),
    ("India", ("new delhi",)), ("Russia", ("moscow",)),
    ("Canada", ("ottawa",)), ("Australia", ("canberra",)),
    ("Brazil", ("brasilia",)), ("Argentina", ("buenos aires",)),
    ("Mexico", ("mexico city",)), ("Egypt", ("cairo",)),
    ("Kenya", ("nairobi",)), ("Nigeria", ("abuja",)),
    ("Turkey", ("ankara",)), ("Greece", ("athens",)),
    ("Norway", ("oslo",)), ("Sweden", ("stockholm",)),
    ("Finland", ("helsinki",)), ("Denmark", ("copenhagen",)),
    ("Poland", ("warsaw",)), ("Austria", ("vienna",)),
    ("Hungary", ("budapest",)), ("Ireland", ("dublin",)),
    ("Peru", ("lima",)), ("Chile", ("santiago",)),
    ("Colombia", ("bogota",)), ("Thailand", ("bangkok",)),
    ("Vietnam", ("hanoi",)), ("South Korea", ("seoul",)),
    ("Pakistan", ("islamabad",)), ("Iran", ("tehran",)),
    ("Saudi Arabia", ("riyadh",)), ("Morocco", ("rabat",)),
    ("Ethiopia", ("addis ababa",)), ("New Zealand", ("wellington",)),
    ("the Czech Republic", ("prague",)), ("Cuba", ("havana",)),
    ("the Philippines", ("manila",)), ("Belgium", ("brussels",)),
    ("Afghanistan", ("kabul",)), ("Iraq", ("baghdad",)),
    ("Venezuela", ("caracas",)), ("Nepal", ("kathmandu",)),
    ("Bangladesh", ("dhaka",)), ("Ghana", ("accra",)),
    ("Romania", ("bucharest",)),
)

_AUTHORS = (
    ("Pride and Prejudice", ("jane austen", "austen")),
    ("Nineteen Eighty-Four", ("george orwell", "orwell", "eric arthur blair")),
    ("Moby-Dick", ("herman melville", "melville")),
    ("War and Peace", ("leo tolstoy", "tolstoy", "lev tolstoy")),
    ("Crime and Punishment", ("fyodor dostoevsky", "dostoevsky",
                              "fyodor dostoyevsky", "dostoyevsky")),
    ("The Great Gatsby", ("f scott fitzgerald", "scott fitzgerald",
                          "fitzgerald")),
    ("To Kill a Mockingbird", ("harper lee", "lee")),
    ("Don Quixote", ("miguel de cervantes", "cervantes",
                     "miguel de cervantes saavedra")),
    ("Frankenstein", ("mary shelley", "shelley", "mary wollstonecraft shelley")),
    ("Dune", ("frank herbert", "herbert")),
    ("The Hobbit", ("j r r tolkien", "tolkien", "jrr tolkien")),
    ("Jane Eyre", ("charlotte bronte", "bronte")),
    ("Wuthering Heights", ("emily bronte", "bronte")),
    ("Great Expectations", ("charles dickens", "dickens")),
    ("Ulysses", ("james joyce", "joyce")),
    ("The Catcher in the Rye", ("j d salinger", "salinger", "jd salinger")),
    ("Brave New World", ("aldous huxley", "huxley")),
    ("Anna Karenina", ("leo tolstoy", "tolstoy", "lev tolstoy")),
    ("Madame Bovary", ("gustave flaubert", "flaubert")),
    ("Les Miserables", ("victor hugo", "hugo")),
    ("One Hundred Years of Solitude", ("gabriel garcia marquez",
                                       "garcia marquez", "marquez")),
    ("The Old Man and the Sea", ("ernest hemingway", "hemingway")),
    ("Lolita", ("vladimir nabokov", "nabokov")),
    ("Dracula", ("bram stoker", "stoker")),
    ("The Picture of Dorian Gray", ("oscar wilde", "wilde")),
    ("Fahrenheit 451", ("ray bradbury", "bradbury")),
    ("The Grapes of Wrath", ("john steinbeck", "steinbeck")),
    ("Things Fall Apart", ("chinua achebe", "achebe")),
    ("Beloved", ("toni morrison", "morrison")),
    ("Gulliver's Travels", ("jonathan swift", "swift")),
    ("Robinson Crusoe", ("daniel defoe", "defoe")),
    ("The Trial", ("franz kafka", "kafka")),
    ("Middlemarch", ("george eliot", "eliot", "mary ann evans")),
    ("Treasure Island", ("robert louis stevenson", "stevenson")),
    ("The Count of Monte Cristo", ("alexandre dumas", "dumas")),
    ("Catch-22", ("joseph heller", "heller")),
    ("Slaughterhouse-Five", ("kurt vonnegut", "vonnegut")),
    ("The Handmaid's Tale", ("margaret atwood", "atwood")),
    ("Heart of Darkness", ("joseph conrad", "conrad")),
    ("The Stranger", ("albert camus", "camus")),
    ("Siddhartha", ("hermann hesse", "hesse")),
    ("The Sun Also Rises", ("ernest hemingway", "hemingway")),
    ("Of Mice and Men", ("john steinbeck", "steinbeck")),
    ("Animal Farm", ("george orwell", "orwell")),
    ("Emma", ("jane austen", "austen")),
    ("Oliver Twist", ("charles dickens", "dickens")),
    ("The Brothers Karamazov", ("fyodor dostoevsky", "dostoevsky",
                                "fyodor dostoyevsky", "dostoyevsky")),
    ("A Tale of Two Cities", ("charles dickens", "dickens")),
    ("The Scarlet Letter", ("nathaniel hawthorne", "hawthorne")),
    ("Little Women", ("louisa may alcott", "alcott")),
)

_INVENTED_ELEMENTS = (
    "veltrium", "marthonium", "caspolium", "drennium", "ulvexium",
    "pravadium", "sorbenium", "tarquilium", "welmium", "yszarium",
    "orvandium", "kelthorium", "brisanium", "fendrium", "noxarium",
    "quenlium", "rastonium", "zephrium", "hollandium", "vessarium",
)

_INVENTED_COUNTRIES = (
    "Veltrania", "Mordavia", "Quessland", "Tarvonia", "Brellmark",
    "Varnistan", "Kalvoria", "Duskmere", "Oristal", "Sarvania",
    "Thessmark", "Ulvania", "Zandoria", "Carvessia", "Norvalia",
    "Estravia", "Gorvannia", "Hollistan", "Vessaria", "Pellatine",
)

_INVENTED_NOVELS = (
    "The Glass Orchard of Minsk", "A Lantern for Sullen Kings",
    "The Cartographer's Ninth Wife", "Beneath the Copper Moons of Ardel",
    "The Salt Clockmaker of Vessary", "Nine Rivers to Kessmere",
    "The Orchard Beneath Halvard", "Letters from the Tin Lighthouse of Obel",
    "The Weaver of Quiet Storms in Tarvonia", "The Porcelain Harvest of Duskmere",
    "Sermons for a Drowned Parish in Veltrania", "The Violinist of Harrowgate Lane",
    "A Grammar of Lost Ferries", "The Moth Archive of Brellmark",
    "Seven Doors in Kessmere", "The Brass Heron of Ulvania",
    "Under the Amber Tides of Loen", "The Quiet Almanac of Doctor Vell",
    "The Last Cartwright of Zandoria", "An Inventory of Salt and Thunder",
)


def _symbol(element):
    return f"What is the chemical symbol of {element}?"


def _number(element):
    return f"What is the atomic number of {element}?"


def _capital(country):
    return f"What is the capital of {country}?"


def _author(title):
    return f"Who wrote the novel {title}?"


KNOWN = tuple(
    [Item("symbol", e, _symbol(e), (s,)) for e, s in _SYMBOLS]
    + [Item("capital", c, _capital(c), a) for c, a in _CAPITALS]
    + [Item("author", t, _author(t), a) for t, a in _AUTHORS]
    + [Item("number", e, _number(e), (n,)) for e, n in _NUMBERS])

FICTITIOUS = tuple(
    [Item("symbol", e, _symbol(e)) for e in _INVENTED_ELEMENTS]
    + [Item("capital", c, _capital(c)) for c in _INVENTED_COUNTRIES]
    + [Item("author", t, _author(t)) for t in _INVENTED_NOVELS]
    + [Item("number", e, _number(e)) for e in _INVENTED_ELEMENTS])


#: Anything that declines to name an answer, wherever it appears. Two
#: teachers agreeing that a country "is fictional" must never become a
#: promoted capital, so a refusal is found anywhere in the reply, not
#: only at its start.
_ABSTAIN = re.compile(
    r"\b(?:unknown|do not know|don t know|not known|no such|"
    r"there is no|there are no|does not exist|doesn t exist|not exist|"
    r"not a real|not an actual|not real|no known|no record|no information|"
    r"not aware|not familiar|not sure|unsure|cannot answer|can t answer|"
    r"cannot determine|unable to|fictional|fictitious|imaginary|made up|"
    r"invented|hypothetical|not a recognized|not recognized|no country|"
    r"no element|no novel|not a known)\b")


def normalize(text: str) -> str:
    """Fold case, accents, punctuation, leading articles and trailing periods."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = text.replace("'", " ")
    words = text.split()
    while words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def is_abstention(answer: str) -> bool:
    """Whether an answer declines to name anything."""
    folded = normalize(answer)
    return not folded or folded == "none" or bool(_ABSTAIN.search(folded))


def is_correct(item: Item, answer: str) -> bool:
    """Whether ``answer`` names one of the item's accepted answers.

    Accepted when the folded answer equals an accepted form, or ends
    with one after a space ("the novelist Jane Austen", "its symbol is
    Au"). A one-word surname is accepted only on its own, so "Taylor
    Swift" does not pass as the author of Gulliver's Travels. Symbols
    and numbers are specific enough to accept at the end of a sentence.
    """
    if item.fictitious or is_abstention(answer):
        return False
    folded = normalize(answer)
    for accepted in item.answers:
        if folded == accepted:
            return True
        tail_ok = " " in accepted or item.category in ("symbol", "number")
        if tail_ok and folded.endswith(" " + accepted):
            return True
    return False


def split(items, half: int) -> tuple:
    """Calibration half (0) or held-out half (1), alternating by position
    within each category, so both halves see every category."""
    seen: dict = {}
    chosen = []
    for item in items:
        index = seen.get(item.category, 0)
        seen[item.category] = index + 1
        if index % 2 == half:
            chosen.append(item)
    return tuple(chosen)
