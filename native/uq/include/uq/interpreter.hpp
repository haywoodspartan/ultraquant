// The thought pipeline, natively - the spine of interpreter/thoughts.py.
//
// This is where the native tier stops being a library and starts
// being the system: text in, a sentence out, a store that changed.
// The parity obligation is at its strictest here, because what a
// pipeline produces IS a sentence - so the gate compares the words,
// not a structure that could be worded two ways.
//
// Scope, stated rather than implied. This unit ports the spine:
//
//   * intent - a statement, a question, an expression, or chat
//   * statements - parsed, stored, spoken back, with polarity, with
//     a revision named aloud, with the near-key adjacency note
//   * questions - the arithmetic and list readers, then the exact
//     recall branch with §11.29's coverage rule, then nearest-held,
//     then the honest unknown
//
// NOT ported here, and so not claimed: polar questions, why, the
// comparative and superlative families, aggregates, history,
// choices, testimony, conjunctions and clause splitting, chains,
// glyphs, goals, URLs and the learning mode. Those arrive with their
// own gates; a gate whose corpus quietly avoids what a tier cannot
// do has measured nothing.
#ifndef UQ_INTERPRETER_HPP
#define UQ_INTERPRETER_HPP

#include <string>

#include "uq/memory.hpp"

namespace uq {

struct Turn {
    std::string intent;     // fact_statement | question | chat
    std::string response;
};

class Session {
public:
    Turn run(const std::string& text);

    Memory& memory() { return memory_; }
    const Memory& memory() const { return memory_; }

private:
    Memory memory_;

    Turn learn_statement(const std::string& text);
    Turn answer_question(const std::string& text);
};

}  // namespace uq

#endif  // UQ_INTERPRETER_HPP
