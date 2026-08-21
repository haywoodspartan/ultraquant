// The arithmetic reader, natively - a port of reason/calculate.py
// under the rule that it must SAY THE SAME THING.
//
// Parity here is checked as strings, not as behaviour-in-spirit:
// the same question must produce the same expression echo, the same
// rendered value, the same refusal sentence and the same rounding
// note. That is a harder target than "works correctly", and it is
// the only target worth having when one tier defines the semantics
// and the other reproduces them.
//
// This is the literal reader - numbers, quantities, percentages,
// powers, roots and lists. Operands that name BELIEFS need the fact
// store and arrive with it.
#ifndef UQ_CALCULATE_HPP
#define UQ_CALCULATE_HPP

#include <string>
#include <vector>

#include "uq/rational.hpp"

namespace uq {

// A number, and the unit it is a number OF ("" for a pure one).
struct Quantity {
    Rational value;
    std::string unit;

    Quantity() = default;
    Quantity(Rational v, std::string u = "") 
        : value(std::move(v)), unit(std::move(u)) {}
};

// One evaluated expression, one enclosure, or one refusal - the
// same shape reason/calculate.py's MathResult carries.
struct MathResult {
    bool present = false;        // false == "not my question"
    std::string shown;
    std::string refusal;
    std::string expression;
    bool fractional = false;
    bool has_bounds = false;
    std::string low;
    std::string high;
    long long rounded_to = -1;   // -1 == no rounding was asked for
    bool was_rounded = false;
    std::string exact_shown;
};

// The English plural fold the router applies, needed here because a
// unit word is recognised after folding: "meters" is a meter.
std::string normalize_token(const std::string& token);

// An exact rational printed the way the Python tier prints it:
// an integer, an exact decimal where the denominator allows one,
// and a fraction where it does not - "1/3", never 0.333...
std::string render_exact(const Rational& value);

// A bare written operand - "200" or "200 meters" - or absent.
bool read_quantity(const std::string& text, Quantity& out);

// "the average of 3, 5 and 10", or absent.
MathResult read_list(const std::string& text);

// The whole reader. A result with present=false means "not my
// question", which is a different thing from a refusal.
MathResult evaluate(const std::string& text);

}  // namespace uq

#endif  // UQ_CALCULATE_HPP
