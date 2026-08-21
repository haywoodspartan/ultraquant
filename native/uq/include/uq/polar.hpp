// The yes/no family, natively - the polar branch of thoughts.py.
//
// Four questions share one door and one discipline. "Is the tower
// steel?" checks a claim against a belief; "is the tower steel or
// iron?" offers the values and lets the store elect one; "is the
// tower taller than the bridge?" compares two numbers; and "is the
// dome city climate temperate?" derives its own subject before it
// can check anything.
//
// The discipline they share is the one worth porting carefully:
// **absence is never No.** A subject the library holds nothing about
// does not get a verdict - it falls through to the hedging machinery
// above, because "no" is reserved for actual contrary belief, held
// either as a different value or as a stored denial. Every early
// return in here is that line being kept.
//
// What is NOT ported, and so not claimed: a comparison side that is
// an expression over BELIEFS ("is the tower height * 2 taller than
// the bridge?") needs the derived-operand reader, which is not in
// this tier yet. Literal expressions ("is 3 * 4 greater than 10?")
// are ported. The gate provokes the unported case freely and counts
// it rather than avoiding it.
#ifndef UQ_POLAR_HPP
#define UQ_POLAR_HPP

#include <string>

#include "uq/memory.hpp"

namespace uq {

// True when this branch OWNS the question, with the sentence in
// `said`. False means "not my question" - which is a different thing
// from a refusal, and the difference is the whole point: a refusal
// is spoken, a decline falls through to the hedge.
bool polar_answer(const std::string& text, const Memory& memory,
                  std::string& said);

// The same branch, with the unported case flagged rather than
// silently answered as though the tiers merely disagreed. `unported`
// is set when a comparison side is an expression over beliefs; the
// gate counts those and bounds the debt, which is a measurement of
// what is missing rather than a silence about it.
bool polar_answer_gapped(const std::string& text, const Memory& memory,
                         std::string& said, bool& unported);

}  // namespace uq

#endif  // UQ_POLAR_HPP
