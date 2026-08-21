// The enumerated-family question forms - superlatives and aggregates.
#ifndef UQ_FORMS_HPP
#define UQ_FORMS_HPP

#include <string>

#include "uq/memory.hpp"

namespace uq {

// Each returns true when it OWNS the question, with the sentence in
// `said`. Returning false means "not my question", which is a
// different thing from refusing.
bool superlative_answer(const std::string& text, const Memory& memory,
                        std::string& said);
bool aggregate_answer(const std::string& text, const Memory& memory,
                      std::string& said);

}  // namespace uq

#endif  // UQ_FORMS_HPP
