// The text primitives every tier above shares.
//
// These were written three times - once in the store, once in the
// spread, once in the interpreter - and three copies of a stopword
// list is not a tidiness problem. §11.29's coverage rule asks
// whether a question's INFORMATIVE tokens are covered by a key, so
// if one copy drifted the tiers would disagree about what a question
// asked, silently, in one branch only. One copy, and a test that
// checks it against the Python tier it was copied from.
#ifndef UQ_TEXT_HPP
#define UQ_TEXT_HPP

#include <set>
#include <string>
#include <vector>

namespace uq {

// Fold a regular English plural onto its singular - the router's
// rule, deliberately conservative: nothing under four characters
// moves, and -ss/-us/-is/-os are left alone.
std::string normalize_token(const std::string& token);

// Lowercased [a-z0-9]+ runs, with NO plural fold. The store's own
// retrieval compares these; the fold belongs to the callers above it.
std::vector<std::string> raw_tokens(const std::string& text);

// Whether a token says anything about what a question is asking.
// Length test and stopword list both, exactly as the Python tier has
// it - the list is generated from it and pinned against it.
bool informative(const std::string& token);

// The informative tokens, folded. A std::set is sorted, which is
// what the Python side asks for wherever the order is observable.
std::set<std::string> folded_tokens(const std::string& text);

// The same tokens in QUESTION order, deduplicated. Order is evidence
// at density, so this is not a convenience.
std::vector<std::string> ordered_tokens(const std::string& text);

}  // namespace uq

#endif  // UQ_TEXT_HPP
