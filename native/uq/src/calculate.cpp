#include "uq/calculate.hpp"

#include <algorithm>
#include <map>
#include <regex>
#include <stdexcept>

namespace uq {
namespace {

// §11.42's unit table, in the exact rationals its float spellings
// denote. The Python tier reads its own table back through
// Fraction(str(factor)) precisely so a definition stays a definition
// rather than becoming the nearest double; these are those readings,
// written out. 1e-06 is here as one millionth because str(1e-06) is
// "1e-06" and Fraction reads that exactly - the one entry where the
// two tiers could have disagreed without either being obviously
// wrong.
struct UnitEntry { const char* name; const char* factor; };

const UnitEntry kLength[] = {
    {"millimeter", "1/1000"}, {"centimeter", "1/100"},
    {"meter", "1"}, {"kilometer", "1000"},
    {"foot", "381/1250"}, {"mile", "201168/125"},
};
const UnitEntry kMass[] = {
    {"milligram", "1/1000000"}, {"gram", "1/1000"},
    {"kilogram", "1"}, {"tonne", "1000"},
};
const UnitEntry kTime[] = {
    {"second", "1"}, {"minute", "60"},
    {"hour", "3600"}, {"day", "86400"},
};

struct Family { const char* name; const UnitEntry* units; std::size_t count; };

const Family kFamilies[] = {
    {"length", kLength, sizeof(kLength) / sizeof(kLength[0])},
    {"mass", kMass, sizeof(kMass) / sizeof(kMass[0])},
    {"time", kTime, sizeof(kTime) / sizeof(kTime[0])},
};

const Family* family_of(const std::string& unit) {
    for (const Family& family : kFamilies) {
        for (std::size_t i = 0; i < family.count; ++i) {
            if (unit == family.units[i].name) return &family;
        }
    }
    return nullptr;
}

Rational factor_of(const Family& family, const std::string& unit) {
    for (std::size_t i = 0; i < family.count; ++i) {
        if (unit == family.units[i].name)
            return Rational::parse(family.units[i].factor);
    }
    throw std::domain_error("unit not in family");
}

bool same_family(const std::string& left, const std::string& right) {
    const Family* a = family_of(left);
    const Family* b = family_of(right);
    return a != nullptr && a == b;
}

class Undefined : public std::runtime_error {
public:
    explicit Undefined(const std::string& why) : std::runtime_error(why) {}
};

class NotArithmetic : public std::runtime_error {
public:
    NotArithmetic() : std::runtime_error("not arithmetic") {}
};

class Irrational : public std::runtime_error {
public:
    Irrational(Rational v, int d)
        : std::runtime_error("irrational"), value(std::move(v)), degree(d) {}
    Rational value;
    int degree;
};

}  // namespace

std::string normalize_token(const std::string& token) {
    // The router's plural fold, reproduced exactly - including the
    // three-character floor and the -ss/-us/-is/-os exceptions,
    // because a unit is recognised only after folding.
    const std::size_t size = token.size();
    if (size < 4) return token;
    auto ends_with = [&token](const std::string& tail) {
        return token.size() >= tail.size()
            && token.compare(token.size() - tail.size(), tail.size(), tail) == 0;
    };
    if (ends_with("ies") && size > 4) return token.substr(0, size - 3) + "y";
    for (const std::string& ending : {"ches", "shes", "xes", "zes", "sses"}) {
        if (ends_with(ending) && size - 2 >= 3)
            return token.substr(0, size - 2);
    }
    if (ends_with("s") && !ends_with("ss") && !ends_with("us")
        && !ends_with("is") && !ends_with("os")) {
        const std::string stripped = token.substr(0, size - 1);
        if (stripped.size() >= 3) return stripped;
    }
    return token;
}

namespace {

std::string fixed_string(const Rational& value, long long places) {
    // A rational printed at exactly `places` decimals. The padding is
    // part of the answer: "to 2 decimal places" is answered in two.
    if (places == 0) return value.floor().str();
    const Rational scaled = value * Rational(BigInt::pow10(places), BigInt(1));
    const BigInt whole = scaled.floor();
    std::string digits = whole.abs().str();
    const std::size_t want = static_cast<std::size_t>(places) + 1;
    if (digits.size() < want)
        digits = std::string(want - digits.size(), '0') + digits;
    const std::string sign = value.negative() ? "-" : "";
    return sign + digits.substr(0, digits.size() - static_cast<std::size_t>(places))
        + "." + digits.substr(digits.size() - static_cast<std::size_t>(places));
}

Rational round_to(const Rational& value, long long places) {
    // Half away from zero, in exact rational arithmetic, so the tie
    // rule is a decision rather than an artefact of how a float
    // happened to land.
    const BigInt scale = BigInt::pow10(places);
    const Rational shifted = value * Rational(scale, BigInt(1));
    BigInt floor = shifted.floor();
    const Rational remainder = shifted - Rational(floor, BigInt(1));
    const Rational half(BigInt(1), BigInt(2));
    if (remainder > half) floor = floor + BigInt(1);
    else if (remainder == half && !value.negative()) floor = floor + BigInt(1);
    return Rational(floor, scale);
}

std::string show(const Quantity& quantity) {
    const std::string text = render_exact(quantity.value);
    return quantity.unit.empty() ? text : text + " " + quantity.unit + "s";
}

// Both values in one unit, plus that unit - or a refusal. The result
// reads in the LARGER of the two, which is the rule §11.42's combine
// path already answers by: one store must not have two voices.
void aligned(const Quantity& left, const Quantity& right,
             Rational& out_left, Rational& out_right, std::string& out_unit) {
    if (left.unit == right.unit) {
        out_left = left.value;
        out_right = right.value;
        out_unit = left.unit;
        return;
    }
    if (left.unit.empty() || right.unit.empty()) {
        throw Undefined("'" + show(left) + "' and '" + show(right)
                        + "' do not agree on a unit, and assuming one "
                          "would be invention");
    }
    if (!same_family(left.unit, right.unit)) {
        throw Undefined("no definition I hold connects " + left.unit
                        + "s and " + right.unit + "s");
    }
    const Family* family = family_of(left.unit);
    const Rational left_factor = factor_of(*family, left.unit);
    const Rational right_factor = factor_of(*family, right.unit);
    const std::string target =
        left_factor >= right_factor ? left.unit : right.unit;
    const Rational target_factor = factor_of(*family, target);
    out_left = left.value * (left_factor / target_factor);
    out_right = right.value * (right_factor / target_factor);
    out_unit = target;
}

Quantity add_quantity(const Quantity& left, const Quantity& right, int sign) {
    Rational a, b;
    std::string unit;
    aligned(left, right, a, b, unit);
    return Quantity(sign > 0 ? a + b : a - b, unit);
}

Quantity mul_quantity(const Quantity& left, const Quantity& right) {
    if (!left.unit.empty() && !right.unit.empty()) {
        throw Undefined("multiplying " + left.unit + "s by " + right.unit
                        + "s would name a unit no definition I hold covers");
    }
    return Quantity(left.value * right.value,
                    left.unit.empty() ? right.unit : left.unit);
}

Quantity div_quantity(const Quantity& left, const Quantity& right) {
    if (right.value.is_zero()) throw std::domain_error("division by zero");
    if (!left.unit.empty() && !right.unit.empty()) {
        if (!same_family(left.unit, right.unit)) {
            throw Undefined("no definition I hold connects " + left.unit
                            + "s and " + right.unit + "s");
        }
        Rational a, b;
        std::string unit;
        aligned(left, right, a, b, unit);
        // A length over a length is a plain ratio, and saying so is
        // more useful than refusing.
        return Quantity(a / b, "");
    }
    if (!right.unit.empty()) {
        throw Undefined("dividing a plain number by " + right.unit
                        + "s would name a unit no definition I hold covers");
    }
    return Quantity(left.value / right.value, left.unit);
}

}  // namespace

std::string render_exact(const Rational& value) {
    if (value.is_integer()) return value.top().str();
    // A denominator whose only prime factors are 2 and 5 has an exact
    // decimal form, and it is printed in full. Any other has none -
    // one third is not 0.333333, and printing that instead would
    // answer a question about thirds with a number that is not the
    // answer - so the fraction itself is the answer.
    BigInt rest = value.bottom();
    long long twos = 0, fives = 0;
    const BigInt two(2), five(5);
    while ((rest % two).is_zero()) { rest = rest / two; ++twos; }
    while ((rest % five).is_zero()) { rest = rest / five; ++fives; }
    if (rest != BigInt(1))
        return value.top().str() + "/" + value.bottom().str();
    return fixed_string(value, std::max(twos, fives));
}

namespace {

Rational exact_root(const Rational& value, int degree, bool& exact) {
    const bool negative = value.negative();
    if (negative && degree % 2 == 0) {
        throw Undefined("no real number raised to the power "
                        + std::to_string(degree) + " gives "
                        + render_exact(value));
    }
    const BigInt top = BigInt::iroot(value.top().abs(), degree);
    const BigInt bottom = BigInt::iroot(value.bottom(), degree);
    if (BigInt::pow(top, degree) != value.top().abs()
        || BigInt::pow(bottom, degree) != value.bottom()) {
        exact = false;
        return Rational();
    }
    exact = true;
    const Rational root(top, bottom);
    return negative ? -root : root;
}

// Exact decimal bounds for an irrational root. A root with no exact
// value still has an exact PLACE, and naming that interval is honest
// in a way 1.4142135623730951 is not.
void enclose(const Rational& value, int degree, long long places,
             Rational& low, Rational& high) {
    const BigInt scale = BigInt::pow10(places);
    const BigInt scaled_top =
        value.top().abs() * BigInt::pow(scale, degree)
        * BigInt::pow(value.bottom(), degree - 1);
    low = Rational(BigInt::iroot(scaled_top, degree),
                   scale * value.bottom());
    const Rational step(BigInt(1), scale);
    // Prove the bracket rather than trusting the scaling: the floor
    // above can land a step low, and an unproved bound is a guess.
    while (low.pow(degree) > value) low = low - step;
    while ((low + step).pow(degree) <= value) low = low + step;
    high = low + step;
}

Rational rounded_root(const Rational& value, int degree, long long places) {
    // Enclose at increasing precision until BOTH bounds round to the
    // same decimal; then that decimal is the correct rounding
    // because two proved bounds agree on it, not because a float
    // looked like it.
    long long extra = places + 3;
    while (true) {
        Rational low, high;
        enclose(value, degree, extra, low, high);
        const Rational down = round_to(low, places);
        const Rational up = round_to(high, places);
        if (down == up) return down;
        extra += 3;
    }
}

const long long kMaxExponent = 1000;

Quantity power_quantity(const Quantity& left, const Quantity& right) {
    if (!right.unit.empty()) {
        throw Undefined("an exponent in " + right.unit
                        + "s is not a number of times");
    }
    if (!right.value.is_integer()) {
        throw Undefined("an exponent that is not a whole number asks for "
                        "a root - ask me for the root itself and I will "
                        "bound it exactly");
    }
    const BigInt whole = right.value.top();
    if (!whole.fits_ll() || std::llabs(whole.to_ll()) > kMaxExponent) {
        throw Undefined("an exponent past " + std::to_string(kMaxExponent)
                        + " names a number with more digits than an "
                          "answer has");
    }
    const long long exponent = whole.to_ll();
    if (left.value.is_zero() && exponent == 0) {
        throw Undefined("zero to the power zero is a convention, not a "
                        "value - different fields choose differently, and "
                        "I will not choose for you");
    }
    if (left.value.is_zero() && exponent < 0) {
        throw std::domain_error("division by zero");
    }
    if (!left.unit.empty() && exponent != 1) {
        throw Undefined("raising " + left.unit + "s to a power would name "
                        "a unit no definition I hold covers");
    }
    return Quantity(left.value.pow(exponent), left.unit);
}

}  // namespace

namespace {

const std::pair<const char*, const char*> kSpokenPowers[] = {
    {R"(\b(?:the\s+)?square\s+roots?\s+of\b)", "sqrt"},
    {R"(\b(?:the\s+)?(?:cube|cubic)\s+roots?\s+of\b)", "cbrt"},
    {R"(\bsquare\s+roots?\b)", "sqrt"},
    {R"(\b(?:cube|cubic)\s+roots?\b)", "cbrt"},
    {R"(\bsqrt\s+of\b)", "sqrt"},
    {R"(\bcbrt\s+of\b)", "cbrt"},
    {R"(\bto\s+the\s+power\s+of\b)", "^"},
    {R"(\braised\s+to\b)", "^"},
    {R"(\bsquared\b)", "^ 2"},
    {R"(\bcubed\b)", "^ 3"},
};

const char* kPrefixes[] = {
    "what is", "what's", "whats", "what does", "how much is",
    "calculate", "compute", "evaluate", "work out", "tell me",
    "equals", "equal",
};

std::string lower_strip(const std::string& text) {
    std::string low;
    low.reserve(text.size());
    for (char c : text) low.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
    const std::string cut = "?!. ";
    std::size_t begin = low.find_first_not_of(cut);
    if (begin == std::string::npos) return "";
    std::size_t end = low.find_last_not_of(cut);
    return low.substr(begin, end - begin + 1);
}

std::string trim(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const std::size_t end = text.find_last_not_of(" \t");
    return text.substr(begin, end - begin + 1);
}

std::string strip_question_impl(const std::string& text) {
    std::string low = lower_strip(text);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const char* prefix : kPrefixes) {
            const std::string lead = std::string(prefix) + " ";
            if (low.compare(0, lead.size(), lead) == 0) {
                low = trim(low.substr(lead.size()));
                changed = true;
            } else if (low == prefix) {
                return "";
            }
        }
    }
    for (const auto& rule : kSpokenPowers) {
        low = std::regex_replace(low, std::regex(rule.first), rule.second);
    }
    return trim(low);
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_lower(char c) { return (c >= 'a' && c <= 'z') || c == '\''; }

// The token regex, hand-rolled: a number, one of ()+-*/%^, or a run
// of letters. Hand-rolled rather than std::regex because this runs
// per question and the shape is trivial - and because the tokeniser
// is the one place where a subtle regex-dialect difference between
// Python and ECMAScript would show up as a parity failure nobody
// could read.
std::vector<std::string> tokenize_raw(const std::string& text) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at < text.size()) {
        const char c = text[at];
        if (is_digit(c)) {
            std::size_t end = at;
            while (end < text.size() && is_digit(text[end])) ++end;
            if (end + 1 < text.size() && text[end] == '.'
                && is_digit(text[end + 1])) {
                ++end;
                while (end < text.size() && is_digit(text[end])) ++end;
            }
            out.push_back(text.substr(at, end - at));
            at = end;
        } else if (std::string("()+-*/%^").find(c) != std::string::npos) {
            out.push_back(std::string(1, c));
            ++at;
        } else if (is_lower(c)) {
            std::size_t end = at;
            while (end < text.size() && is_lower(text[end])) ++end;
            out.push_back(text.substr(at, end - at));
            at = end;
        } else {
            ++at;
        }
    }
    return out;
}

std::string squeeze(const std::string& text) {
    std::string out;
    for (char c : text) if (!std::isspace(static_cast<unsigned char>(c)))
        out.push_back(c);
    return out;
}

bool is_number(const std::string& token) {
    if (token.empty() || !is_digit(token[0])) return false;
    bool dot = false;
    for (std::size_t i = 1; i < token.size(); ++i) {
        if (token[i] == '.') {
            if (dot) return false;
            dot = true;
        } else if (!is_digit(token[i])) {
            return false;
        }
    }
    return true;
}

bool is_unit_word(const std::string& word) {
    return family_of(normalize_token(word)) != nullptr;
}

}  // namespace
