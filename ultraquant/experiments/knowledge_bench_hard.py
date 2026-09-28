"""The harder tier: facts that are certain, but not the ones everyone knows.

§11.130's benchmark chose facts beyond dispute, and all three local
teachers knew every one, so coverage and precision sat at their
ceiling and the filter's real trade-off never showed. This tier keeps
the rule that every answer must be beyond dispute, and drops the rule
that it must be famous:
- capitals of US states, of Canadian provinces and territories, and of
  Australian states and territories;
- the atomic numbers and symbols of the heavier and rarer elements;
- authors of well-established but less famous novels.
Where teachers disagree, the filter has to choose, and that choice is
what this tier measures.

**What was left out, and why.** "Invisible Man" is Ellison's, but "The
Invisible Man" is Wells's. There is more than one "Blindness", and more
than one "The Emigrants". So titles that name more than one book are
out, and US states are asked as "the US state of Washington", never as
bare names a country shares. The question form is fixed by
``knowledge_bench``'s templates.

**One edge case kept on purpose.** Nobelium's symbol is "No". The
§11.133 rule that an answer containing a negation is no position will
refuse it however many teachers agree. It stays in, so the cost of
that rule is measured instead of hidden.

The invented subjects follow the categories. Each name was checked by
hand against the real ones nearest it, after §11.130's "Mordavia" sat
one letter from the real Mordovia.
"""

from __future__ import annotations

from ultraquant.experiments.knowledge_bench import Item

__all__ = ["KNOWN", "FICTITIOUS"]

_US = (
    ("Alabama", "montgomery"), ("Alaska", "juneau"), ("Arizona", "phoenix"),
    ("Arkansas", "little rock"), ("California", "sacramento"),
    ("Colorado", "denver"), ("Connecticut", "hartford"), ("Delaware", "dover"),
    ("Florida", "tallahassee"), ("Georgia", "atlanta"), ("Hawaii", "honolulu"),
    ("Idaho", "boise"), ("Illinois", "springfield"),
    ("Indiana", "indianapolis"), ("Iowa", "des moines"), ("Kansas", "topeka"),
    ("Kentucky", "frankfort"), ("Louisiana", "baton rouge"),
    ("Maine", "augusta"), ("Maryland", "annapolis"),
    ("Massachusetts", "boston"), ("Michigan", "lansing"),
    ("Minnesota", "saint paul", "st paul"), ("Mississippi", "jackson"),
    ("Missouri", "jefferson city"), ("Montana", "helena"),
    ("Nebraska", "lincoln"), ("Nevada", "carson city"),
    ("New Hampshire", "concord"), ("New Jersey", "trenton"),
    ("New Mexico", "santa fe"), ("New York", "albany"),
    ("North Carolina", "raleigh"), ("North Dakota", "bismarck"),
    ("Ohio", "columbus"), ("Oklahoma", "oklahoma city"), ("Oregon", "salem"),
    ("Pennsylvania", "harrisburg"), ("Rhode Island", "providence"),
    ("South Carolina", "columbia"), ("South Dakota", "pierre"),
    ("Tennessee", "nashville"), ("Texas", "austin"),
    ("Utah", "salt lake city"), ("Vermont", "montpelier"),
    ("Virginia", "richmond"), ("Washington", "olympia"),
    ("West Virginia", "charleston"), ("Wisconsin", "madison"),
    ("Wyoming", "cheyenne"),
)

_CANADA = (
    ("Ontario", "toronto"), ("Quebec", "quebec city", "quebec"),
    ("British Columbia", "victoria"), ("Alberta", "edmonton"),
    ("Saskatchewan", "regina"), ("Manitoba", "winnipeg"),
    ("Nova Scotia", "halifax"), ("New Brunswick", "fredericton"),
    ("Prince Edward Island", "charlottetown"),
    ("Newfoundland and Labrador", "st john s", "saint john s"),
    ("Yukon", "whitehorse"), ("the Northwest Territories", "yellowknife"),
    ("Nunavut", "iqaluit"),
)

_AUSTRALIA = (
    ("New South Wales", "sydney"), ("Victoria", "melbourne"),
    ("Queensland", "brisbane"), ("Western Australia", "perth"),
    ("South Australia", "adelaide"), ("Tasmania", "hobart"),
    ("the Northern Territory", "darwin"),
    ("the Australian Capital Territory", "canberra"),
)

_HEAVY_NUMBERS = (
    ("technetium", "43"), ("promethium", "61"), ("europium", "63"),
    ("gadolinium", "64"), ("terbium", "65"), ("dysprosium", "66"),
    ("holmium", "67"), ("erbium", "68"), ("thulium", "69"),
    ("ytterbium", "70"), ("lutetium", "71"), ("hafnium", "72"),
    ("tantalum", "73"), ("rhenium", "75"), ("osmium", "76"),
    ("iridium", "77"), ("polonium", "84"), ("astatine", "85"),
    ("francium", "87"), ("radium", "88"), ("actinium", "89"),
    ("thorium", "90"), ("protactinium", "91"), ("neptunium", "93"),
    ("americium", "95"), ("curium", "96"), ("berkelium", "97"),
    ("californium", "98"), ("einsteinium", "99"), ("fermium", "100"),
    ("mendelevium", "101"), ("nobelium", "102"), ("lawrencium", "103"),
    ("rutherfordium", "104"), ("dubnium", "105"), ("seaborgium", "106"),
    ("bohrium", "107"), ("hassium", "108"), ("meitnerium", "109"),
    ("darmstadtium", "110"), ("roentgenium", "111"),
    ("copernicium", "112"), ("nihonium", "113"), ("flerovium", "114"),
    ("moscovium", "115"), ("livermorium", "116"), ("tennessine", "117"),
    ("oganesson", "118"),
)

_RARE_SYMBOLS = (
    ("niobium", "nb"), ("molybdenum", "mo"), ("ruthenium", "ru"),
    ("rhodium", "rh"), ("palladium", "pd"), ("cadmium", "cd"),
    ("indium", "in"), ("tellurium", "te"), ("caesium", "cs"),
    ("lanthanum", "la"), ("cerium", "ce"), ("praseodymium", "pr"),
    ("neodymium", "nd"), ("samarium", "sm"), ("europium", "eu"),
    ("gadolinium", "gd"), ("terbium", "tb"), ("dysprosium", "dy"),
    ("holmium", "ho"), ("erbium", "er"), ("thulium", "tm"),
    ("ytterbium", "yb"), ("lutetium", "lu"), ("hafnium", "hf"),
    ("tantalum", "ta"), ("rhenium", "re"), ("osmium", "os"),
    ("iridium", "ir"), ("thallium", "tl"), ("polonium", "po"),
    ("astatine", "at"), ("francium", "fr"), ("radium", "ra"),
    ("actinium", "ac"), ("thorium", "th"), ("protactinium", "pa"),
    ("neptunium", "np"), ("americium", "am"), ("curium", "cm"),
    ("berkelium", "bk"), ("californium", "cf"), ("einsteinium", "es"),
    ("fermium", "fm"), ("mendelevium", "md"), ("nobelium", "no"),
    ("lawrencium", "lr"), ("rutherfordium", "rf"), ("dubnium", "db"),
    ("seaborgium", "sg"), ("bohrium", "bh"),
)

_AUTHORS = (
    ("The Master and Margarita", ("mikhail bulgakov", "bulgakov")),
    ("The Tin Drum", ("gunter grass", "gunther grass", "grass")),
    ("Pedro Paramo", ("juan rulfo", "rulfo")),
    ("The Leopard", ("giuseppe tomasi di lampedusa", "tomasi di lampedusa",
                     "lampedusa")),
    ("The Name of the Rose", ("umberto eco", "eco")),
    ("Invisible Cities", ("italo calvino", "calvino")),
    ("The Magic Mountain", ("thomas mann", "mann")),
    ("Steppenwolf", ("hermann hesse", "hesse")),
    ("The Good Soldier Svejk", ("jaroslav hasek", "hasek")),
    ("Kristin Lavransdatter", ("sigrid undset", "undset")),
    ("Fathers and Sons", ("ivan turgenev", "turgenev")),
    ("Oblomov", ("ivan goncharov", "goncharov")),
    ("Dead Souls", ("nikolai gogol", "gogol")),
    ("The Red and the Black", ("stendhal",)),
    ("Germinal", ("emile zola", "zola")),
    ("Candide", ("voltaire",)),
    ("Journey to the End of the Night", ("louis ferdinand celine", "celine")),
    ("Nostromo", ("joseph conrad", "conrad")),
    ("The Moonstone", ("wilkie collins", "collins")),
    ("Vanity Fair", ("william makepeace thackeray", "thackeray")),
    ("Tess of the d'Urbervilles", ("thomas hardy", "hardy")),
    ("The Way of All Flesh", ("samuel butler", "butler")),
    ("Tristram Shandy", ("laurence sterne", "sterne")),
    ("Clarissa", ("samuel richardson", "richardson")),
    ("Tom Jones", ("henry fielding", "fielding")),
    ("Ivanhoe", ("walter scott", "sir walter scott", "scott")),
    ("The Last of the Mohicans", ("james fenimore cooper", "cooper")),
    ("Uncle Tom's Cabin", ("harriet beecher stowe", "stowe")),
    ("The Awakening", ("kate chopin", "chopin")),
    ("Babbitt", ("sinclair lewis", "lewis")),
    ("Native Son", ("richard wright", "wright")),
    ("Their Eyes Were Watching God", ("zora neale hurston", "hurston")),
    ("Wide Sargasso Sea", ("jean rhys", "rhys")),
    ("The Remains of the Day", ("kazuo ishiguro", "ishiguro")),
    ("Midnight's Children", ("salman rushdie", "rushdie")),
    ("Disgrace", ("j m coetzee", "coetzee", "jm coetzee")),
    ("The God of Small Things", ("arundhati roy", "roy")),
    ("Snow Country", ("yasunari kawabata", "kawabata")),
    ("The Tale of Genji", ("murasaki shikibu", "murasaki", "lady murasaki")),
    ("Dream of the Red Chamber", ("cao xueqin",)),
    ("Season of Migration to the North", ("tayeb salih", "salih")),
    ("Palace Walk", ("naguib mahfouz", "mahfouz")),
)

# Invented, and checked against their nearest real names by hand.
_INVENTED_PROVINCES = (
    "the US state of Kessaway", "the US state of North Tarrow",
    "the US state of Wendalia", "the US state of Morrowand",
    "the Canadian province of Vellmark", "the Canadian province of Oskarra",
    "the Australian state of Brellingham", "the Australian state of Quorrin",
    "the US state of Pellatoba", "the US state of East Dunmarrow",
)
_INVENTED_HEAVY = (
    "zorbenium", "quellium", "varnadium", "trevellium", "oskarium",
    "brandelium", "ulquorium", "penthelium", "marrowium", "kestrium",
)
_INVENTED_TITLES = (
    "The Lanternmaker of Oskarra", "Nine Winters in Vellmark",
    "The Salt Archive of Kessaway", "A Ferry for the Quorrin Dead",
    "The Brass Orchard of Wendalia", "Letters to the Morrowand Glassblower",
    "The Cartographer of East Dunmarrow", "Ashes over Brellingham",
    "The Tenth Bell of Pellatoba", "An Almanac of Quiet Trevellium",
)


def _capital(place):
    return f"What is the capital of {place}?"


KNOWN = tuple(
    [Item("capital", f"the US state of {s}", _capital(f"the US state of {s}"),
          tuple(a)) for s, *a in _US]
    + [Item("capital", p, _capital(p), tuple(a)) for p, *a in _CANADA]
    + [Item("capital", p, _capital(p), tuple(a)) for p, *a in _AUSTRALIA]
    + [Item("number", e, f"What is the atomic number of {e}?", (n,))
       for e, n in _HEAVY_NUMBERS]
    + [Item("symbol", e, f"What is the chemical symbol of {e}?", (s,))
       for e, s in _RARE_SYMBOLS]
    + [Item("author", t, f"Who wrote the novel {t}?", a) for t, a in _AUTHORS])

FICTITIOUS = tuple(
    [Item("capital", p, _capital(p)) for p in _INVENTED_PROVINCES]
    + [Item("number", e, f"What is the atomic number of {e}?")
       for e in _INVENTED_HEAVY]
    + [Item("symbol", e, f"What is the chemical symbol of {e}?")
       for e in _INVENTED_HEAVY]
    + [Item("author", t, f"Who wrote the novel {t}?") for t in _INVENTED_TITLES])
