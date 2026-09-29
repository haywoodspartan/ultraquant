"""The dictionary: Open English WordNet as a paged lexicon of its own.

``python -m ultraquant.lexicon.build --home DIR`` imports the release into
``DIR/lexicon``; :class:`Lexicon` reads it a page at a time. The build
submodule is deliberately not imported here, so ``ultraquant.lexicon.build``
stays the module and ``-m`` runs it cleanly.
"""

from ultraquant.lexicon.lexicon import Lexicon, normalize_word

__all__ = ["Lexicon", "normalize_word"]
