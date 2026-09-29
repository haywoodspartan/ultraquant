# Open English WordNet (2025+ edition)

`english-wordnet-2025-plus-json.zip` is the unmodified JSON release of the
**Open English WordNet**, 2025+ edition, by the Open English Wordnet team:
<https://github.com/globalwordnet/english-wordnet>.

- **Size and hash:** 11,298,794 bytes, sha256
  `8832b8fa26a14c0ba8c99bb1ef8db6f9e122a6d9193b65ca9e0ef572580fee7e`.
- **Licenses:**
  - [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/);
  - the [WordNet License](https://wordnet.princeton.edu/license-and-commercial-use),
    since it is derived from **Princeton WordNet** (Princeton University).

UltraQuant does not change the release. `python -m ultraquant.lexicon.build`
reorganizes it into the library's dictionary (`<home>/lexicon/`), with these
differences:
- the words are re-indexed into hash-addressed pages;
- homograph keys ("n-1", "n-2") are read as their part of speech;
- inverse "hyponym" and "instance" links are added.

Every answer drawn from it ends with its credit line.
