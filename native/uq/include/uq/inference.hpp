// Spreading-activation inference, natively - reason/inference.py.
//
// The hardest thing to port faithfully in this system, because its
// correctness lives in ORDER as much as in arithmetic: which facts
// the index surfaces for which probe, which lineage reaches a node
// first, which of several equally-strong convergences is chosen.
// Python dicts preserve insertion order and the spread relies on
// that, so the native tier uses insertion-ordered containers rather
// than std::map wherever the Python side iterates a dict.
//
// The activation strengths are powers of one half, which are exact
// in binary floating point - so the one place a port would normally
// drift, it cannot.
#ifndef UQ_INFERENCE_HPP
#define UQ_INFERENCE_HPP

#include <string>
#include <utility>
#include <vector>

#include "uq/memory.hpp"

namespace uq {

// One derived answer and the evidence trail that produced it.
struct Inference {
    bool present = false;
    std::string answer;
    std::vector<std::pair<std::string, std::string>> premises;
    double confidence = 0.0;
    std::string kind;
    bool has_conclusion = false;
    std::string conclusion_key;
    std::string conclusion_value;
    bool negated = false;

    // The answer with its premises named - inferred, never asserted.
    std::string describe() const;
};

// The fact that would let a refused question converge, or absent.
struct MissingPremise {
    bool present = false;
    std::string premise_key;
    std::string via_key;
    std::string via_value;
    std::string original;
};

Inference infer(const std::string& text, const Memory& memory);
MissingPremise missing_premise(const std::string& text,
                               const Memory& memory);

}  // namespace uq

#endif  // UQ_INFERENCE_HPP
