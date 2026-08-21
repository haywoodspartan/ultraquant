#include "uq/calculate.hpp"

#include "uq/text.hpp"

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

bool unit_family(const std::string& unit, std::string& family) {
    const Family* found = family_of(unit);
    if (found == nullptr) return false;
    family = found->name;
    return true;
}

double unit_factor(const std::string& unit) {
    const Family* found = family_of(unit);
    if (found == nullptr) return 1.0;
    for (std::size_t i = 0; i < found->count; ++i) {
        if (unit == found->units[i].name) {
            const Rational exact = Rational::parse(found->units[i].factor);
            return std::stod(exact.top().str())
                 / std::stod(exact.bottom().str());
        }
    }
    return 1.0;
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

namespace {

// One item of the read structure. The Python tier carries tuples;
// here a small tagged struct says the same thing.
struct Item {
    enum Kind { kOp, kNumber, kPercent, kRoot } kind;
    std::string text;     // operator symbol, number text, or root word
    std::string unit;     // numbers only
    std::string echo;     // operators only, when it differs ("of")
};

const std::map<std::string, std::string>& word_ops() {
    static const std::map<std::string, std::string> table = {
        {"plus", "+"}, {"minus", "-"}, {"times", "*"}, {"over", "/"},
        {"divided", "/"}, {"multiplied", "*"},
    };
    return table;
}

int root_degree(const std::string& word) {
    if (word == "sqrt") return 2;
    if (word == "cbrt") return 3;
    return 0;
}

// Structure only: nothing is looked up here. Returns false for "not
// my question"; throws Undefined for a dangling percentage, which is
// an arithmetic question with no answer rather than a non-question.
bool read_structure(const std::string& text, std::vector<Item>& items) {
    const std::vector<std::string> raw = tokenize_raw(text);
    if (raw.empty()) return false;
    std::string joined;
    for (const std::string& token : raw) joined += token;
    if (joined != squeeze(text)) return false;   // something untokenizable

    bool dangling_percent = false;
    std::size_t index = 0;
    while (index < raw.size()) {
        const std::string& token = raw[index];
        if (is_number(token)) {
            const std::string following =
                index + 1 < raw.size() ? raw[index + 1] : std::string();
            if (following == "%" || following == "percent") {
                // A percentage is a number over a hundred, and "of"
                // is the multiplication it is waiting for. Both
                // halves are structural: nothing downstream is a
                // special case for percents.
                items.push_back({Item::kPercent, token, "", ""});
                ++index;
                if (index + 1 < raw.size()
                    && (raw[index + 1] == "of" || raw[index + 1] == "off")) {
                    items.push_back({Item::kOp, "*", "", "of"});
                    ++index;
                } else {
                    dangling_percent = true;
                }
                ++index;
                continue;
            }
            items.push_back({Item::kNumber, token, "", ""});
        } else if (token == "%") {
            return false;   // a sign with no number in front of it
        } else if (root_degree(token) != 0) {
            items.push_back({Item::kRoot, token, "", ""});
        } else if (token == "^") {
            items.push_back({Item::kOp, "^", "", ""});
        } else if (token.size() == 1
                   && std::string("()+-*/").find(token[0]) != std::string::npos) {
            items.push_back({Item::kOp, token, "", ""});
        } else {
            const auto found = word_ops().find(token);
            if (found == word_ops().end()) return false;   // a word: not ours
            items.push_back({Item::kOp, found->second, "", ""});
            if ((token == "divided" || token == "multiplied")
                && index + 1 < raw.size() && raw[index + 1] == "by") {
                ++index;
            }
        }
        ++index;
    }

    bool has_operator = false;
    bool has_operand = false;
    for (const Item& item : items) {
        if (item.kind == Item::kOp
            && std::string("+-*/^").find(item.text[0]) != std::string::npos)
            has_operator = true;
        if (item.kind == Item::kRoot) has_operator = true;
        if (item.kind != Item::kOp) has_operand = true;
    }
    if (!has_operator || !has_operand) return false;
    if (dangling_percent) {
        // "300 + 20%" is a convention, not an arithmetic: twenty per
        // cent OF WHAT is a question the sentence does not answer,
        // and picking the left operand for the speaker would be
        // guessing in the one branch that exists to be exact.
        throw Undefined("a percentage needs something to be a percentage "
                        "of - '20% of 300' has an answer, '300 + 20%' has "
                        "a convention");
    }
    return true;
}

}  // namespace

namespace {

// A parser token: either an operator/paren/root word, or a resolved
// operand. The Python tier mixes strings and Quantities in one list;
// the same shape, typed.
struct Token {
    bool is_quantity = false;
    Quantity quantity;
    std::string text;
};

class SyntaxTrouble : public std::runtime_error {
public:
    SyntaxTrouble() : std::runtime_error("syntax") {}
};

class Reader {
public:
    explicit Reader(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

    std::size_t at() const { return at_; }
    std::size_t size() const { return tokens_.size(); }

    Quantity expr() {
        Quantity value = term();
        while (peek_is("+") || peek_is("-")) {
            const std::string op = take().text;
            value = add_quantity(value, term(), op == "+" ? 1 : -1);
        }
        return value;
    }

private:
    std::vector<Token> tokens_;
    std::size_t at_ = 0;

    bool peek_is(const std::string& what) const {
        return at_ < tokens_.size() && !tokens_[at_].is_quantity
            && tokens_[at_].text == what;
    }
    bool at_end() const { return at_ >= tokens_.size(); }
    const Token& take() { return tokens_[at_++]; }

    Quantity term() {
        Quantity value = unary();
        while (peek_is("*") || peek_is("/")) {
            const std::string op = take().text;
            const Quantity right = unary();
            value = op == "*" ? mul_quantity(value, right)
                              : div_quantity(value, right);
        }
        return value;
    }

    Quantity unary() {
        if (peek_is("+") || peek_is("-")) {
            const std::string op = take().text;
            const Quantity value = unary();
            return op == "-" ? Quantity(-value.value, value.unit) : value;
        }
        return power();
    }

    // atom ^ unary, right-associative. The sign sits OUTSIDE the
    // power, which is what "-2^2 = -4" means, and the right operand
    // is a unary so "2^-3" reads.
    Quantity power() {
        Quantity value = atom();
        if (peek_is("^")) {
            take();
            return power_quantity(value, unary());
        }
        return value;
    }

    Quantity atom() {
        if (at_end()) throw SyntaxTrouble();
        if (tokens_[at_].is_quantity) return take().quantity;
        const std::string text = tokens_[at_].text;
        const int degree = root_degree(text);
        if (degree != 0) {
            take();
            // A unary, not an atom: "sqrt -4" must reach the root to
            // be refused there, rather than dying as a syntax error
            // and falling through as though it were not arithmetic.
            const Quantity inner = unary();
            if (!inner.unit.empty()) {
                throw Undefined("the root of a length in " + inner.unit
                                + "s would name a unit no definition I "
                                  "hold covers");
            }
            bool exact = false;
            const Rational root = exact_root(inner.value, degree, exact);
            if (exact) return Quantity(root, "");
            throw Irrational(inner.value, degree);
        }
        if (text == "(") {
            take();
            const Quantity value = expr();
            if (!peek_is(")")) throw SyntaxTrouble();
            take();
            return value;
        }
        throw SyntaxTrouble();
    }
};

bool is_unary_at(const std::vector<std::string>& echoes, long long index) {
    if (index < 0) return false;
    const std::string& token = echoes[static_cast<std::size_t>(index)];
    if (token != "+" && token != "-") return false;
    if (index == 0) return true;
    const std::string& before = echoes[static_cast<std::size_t>(index - 1)];
    return (before.size() == 1
            && std::string("(+-*/^").find(before[0]) != std::string::npos)
        || root_degree(before) != 0;
}

// The expression as it was READ, spaced the way people write it -
// which is what makes precedence checkable at a glance.
std::string echo_of(const std::vector<std::string>& echoes) {
    std::string out;
    for (std::size_t index = 0; index < echoes.size(); ++index) {
        const std::string& token = echoes[index];
        const std::string before = index ? echoes[index - 1] : std::string();
        const bool hugs = out.empty() || before == "(" || token == ")"
            || is_unary_at(echoes, static_cast<long long>(index) - 1);
        out += hugs ? token : " " + token;
    }
    return out;
}

}  // namespace

namespace {

const std::pair<const char*, long long> kRoundingRules[] = {
    {R"([, ]*(?:rounded\s+)?to\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:decimal\s+)?places?\s*$)",
     -1},
    {R"([, ]*(?:rounded\s+)?to\s+the\s+nearest\s+(?:whole\s+number|integer)\s*$)",
     0},
};

const char kUnsupportedRounding[] =
    R"([, ]*(?:rounded\s+)?to\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:significant\s+figures?|sig\s*figs?)\s*$|[, ]*(?:rounded\s+)?to\s+the\s+nearest\s+(?:ten|hundred|thousand|tenth|hundredth)\s*$)";

long long word_places(const std::string& word) {
    static const std::map<std::string, long long> table = {
        {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4}, {"five", 5},
        {"six", 6}, {"seven", 7}, {"eight", 8}, {"nine", 9}, {"ten", 10},
    };
    const auto found = table.find(word);
    if (found != table.end()) return found->second;
    return std::stoll(word);
}

bool starts_with(const std::string& text, const std::string& lead) {
    return text.size() >= lead.size()
        && text.compare(0, lead.size(), lead) == 0;
}

}  // namespace

MathResult evaluate(const std::string& text) {
    MathResult out;
    std::string stripped = strip_question_impl(text);
    if (stripped.empty()) return out;

    long long places = -1;
    for (const auto& rule : kRoundingRules) {
        std::smatch found;
        const std::regex pattern(rule.first);
        if (std::regex_search(stripped, found, pattern)) {
            places = rule.second >= 0 ? rule.second
                                      : word_places(found[1].str());
            stripped = trim(stripped.substr(
                0, static_cast<std::size_t>(found.position(0))));
            break;
        }
    }
    if (places < 0) {
        std::smatch found;
        if (std::regex_search(stripped, found,
                              std::regex(kUnsupportedRounding))) {
            out.present = true;
            out.refusal = "I can round to a number of decimal places or "
                          "to the nearest whole number; that is a "
                          "different rule and I do not have it";
            return out;
        }
    }
    // A polar lead asks whether something HOLDS, which is a different
    // question and belongs to the branch that owns it.
    if (starts_with(stripped, "is ") || starts_with(stripped, "are ")
        || starts_with(stripped, "was ") || starts_with(stripped, "were "))
        return out;

    std::vector<Item> items;
    try {
        if (!read_structure(stripped, items)) return out;
    } catch (const Undefined& refusal) {
        out.present = true;
        out.refusal = refusal.what();
        return out;
    }

    std::vector<Token> tokens;
    std::vector<std::string> echoes;
    for (const Item& item : items) {
        if (item.kind == Item::kOp) {
            Token token;
            token.text = item.text;
            tokens.push_back(token);
            echoes.push_back(item.echo.empty() ? item.text : item.echo);
        } else if (item.kind == Item::kRoot) {
            Token token;
            token.text = item.text;
            tokens.push_back(token);
            echoes.push_back(item.text);
        } else if (item.kind == Item::kPercent) {
            Token token;
            token.is_quantity = true;
            token.quantity = Quantity(Rational::parse(item.text)
                                          / Rational(100), "");
            tokens.push_back(token);
            echoes.push_back(item.text + "%");
        } else {
            Token token;
            token.is_quantity = true;
            token.quantity = Quantity(Rational::parse(item.text), item.unit);
            tokens.push_back(token);
            echoes.push_back(show(token.quantity));
        }
    }

    bool any_operator = false;
    for (const Token& token : tokens) {
        if (token.is_quantity) continue;
        if (token.text.size() == 1
            && std::string("+-*/^").find(token.text[0]) != std::string::npos)
            any_operator = true;
        if (root_degree(token.text) != 0) any_operator = true;
    }
    if (!any_operator) return out;   // a bare number is not a calculation

    Reader reader(tokens);
    Quantity value;
    try {
        value = reader.expr();
    } catch (const Irrational& irrational) {
        out.present = true;
        out.expression = echo_of(echoes);
        if (places >= 0) {
            const Rational rounded = rounded_root(irrational.value,
                                                  irrational.degree, places);
            out.shown = fixed_string(rounded, places);
            out.rounded_to = places;
            out.was_rounded = true;
            out.exact_shown = "irrational";
            return out;
        }
        Rational low, high;
        enclose(irrational.value, irrational.degree, 9, low, high);
        out.has_bounds = true;
        out.low = render_exact(low);
        out.high = render_exact(high);
        return out;
    } catch (const Undefined& refusal) {
        out.present = true;
        out.refusal = refusal.what();
        out.expression = echo_of(echoes);
        return out;
    } catch (const std::domain_error&) {
        out.present = true;
        out.refusal = "division by zero is undefined - there is no number "
                      "to give you";
        out.expression = echo_of(echoes);
        return out;
    } catch (const SyntaxTrouble&) {
        return out;
    }
    if (reader.at() != reader.size()) return out;

    out.present = true;
    out.expression = echo_of(echoes);
    if (places >= 0) {
        const Rational exact = value.value;
        const Rational rounded = round_to(exact, places);
        const std::string padded = fixed_string(rounded, places);
        out.shown = value.unit.empty() ? padded
                                       : padded + " " + value.unit + "s";
        out.rounded_to = places;
        out.was_rounded = !(rounded == exact);
        out.exact_shown = render_exact(exact);
        return out;
    }
    out.shown = show(value);
    out.fractional = render_exact(value.value).find('/') != std::string::npos;
    return out;
}

bool read_quantity(const std::string& text, Quantity& out) {
    // Deliberately narrow - a number, optionally followed by one
    // known unit, and nothing else - because anything looser would
    // start reading keys, and keys are the caller's business.
    const std::vector<std::string> tokens = tokenize_raw(lower_strip(text));
    if (tokens.empty() || !is_number(tokens[0])) return false;
    if (tokens.size() == 1) {
        out = Quantity(Rational::parse(tokens[0]), "");
        return true;
    }
    if (tokens.size() == 2 && is_unit_word(tokens[1])) {
        out = Quantity(Rational::parse(tokens[0]), normalize_token(tokens[1]));
        return true;
    }
    return false;
}

namespace {

const std::map<std::string, std::string>& list_words() {
    static const std::map<std::string, std::string> table = {
        {"sum", "sum"}, {"total", "sum"},
        {"average", "average"}, {"mean", "average"},
        {"largest", "largest"}, {"biggest", "largest"},
        {"maximum", "largest"}, {"max", "largest"}, {"greatest", "largest"},
        {"smallest", "smallest"}, {"minimum", "smallest"},
        {"min", "smallest"}, {"least", "smallest"},
    };
    return table;
}

std::vector<std::string> split_items(const std::string& body) {
    // Python splits on "," or the word "and"; the same split, done
    // by hand so the word boundary is explicit.
    std::vector<std::string> pieces;
    std::string current;
    std::size_t at = 0;
    while (at < body.size()) {
        if (body[at] == ',') {
            pieces.push_back(current);
            current.clear();
            ++at;
            continue;
        }
        const bool boundary_before = at == 0
            || !is_lower(body[at - 1]);
        if (boundary_before && body.compare(at, 3, "and") == 0
            && (at + 3 >= body.size() || !is_lower(body[at + 3]))) {
            pieces.push_back(current);
            current.clear();
            at += 3;
            continue;
        }
        current.push_back(body[at]);
        ++at;
    }
    pieces.push_back(current);
    std::vector<std::string> out;
    for (const std::string& piece : pieces) {
        const std::string trimmed = trim(piece);
        if (!trimmed.empty()) out.push_back(trimmed);
    }
    return out;
}

}  // namespace

MathResult read_list(const std::string& text) {
    MathResult out;
    const std::string stripped = strip_question_impl(text);
    std::smatch found;
    std::string pattern = R"RX(^(?:the\s+)?()RX";
    std::vector<std::string> names;
    for (const auto& entry : list_words()) names.push_back(entry.first);
    std::sort(names.begin(), names.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() != b.size() ? a.size() > b.size() : a < b;
              });
    for (std::size_t i = 0; i < names.size(); ++i)
        pattern += (i ? "|" : "") + names[i];
    pattern += R"RX()\s+of\s+(.+)$)RX";
    if (!std::regex_match(stripped, found, std::regex(pattern))) return out;

    const std::string operation = list_words().at(found[1].str());
    std::string body = trim(found[2].str());
    while (!body.empty()
           && (body.back() == '?' || body.back() == '.')) body.pop_back();
    const std::vector<std::string> pieces = split_items(body);
    if (pieces.size() < 2) return out;   // one item is not a list

    std::vector<Quantity> items;
    for (const std::string& piece : pieces) {
        Quantity item;
        if (!read_quantity(piece, item)) return out;
        items.push_back(item);
    }
    try {
        // The running total is computed FIRST, even for largest and
        // smallest, because the Python tier does - and that ordering
        // is visible from outside: on a mixed-unit list the refusal
        // names the units in the order the failing ADDITION met them,
        // not the order a comparison would. Parity checked as strings
        // is what turned that into a caught bug rather than a
        // difference nobody would ever look for.
        Quantity total = items[0];
        for (std::size_t i = 1; i < items.size(); ++i)
            total = add_quantity(total, items[i], 1);
        Quantity value;
        if (operation == "largest" || operation == "smallest") {
            std::vector<Quantity> lined;
            for (const Quantity& item : items) {
                Rational a, b;
                std::string unit;
                aligned(item, items[0], a, b, unit);
                lined.push_back(Quantity(a, unit));
            }
            Quantity best = lined[0];
            for (const Quantity& item : lined) {
                const bool wins = operation == "largest"
                    ? item.value > best.value : item.value < best.value;
                if (wins) best = item;   // ties keep the first, as max() does
            }
            value = best;
        } else {
            value = operation == "average"
                ? Quantity(total.value / Rational(
                      static_cast<long long>(items.size())), total.unit)
                : total;
        }
        std::string spoken;
        for (std::size_t i = 0; i + 1 < items.size(); ++i)
            spoken += (i ? ", " : "") + show(items[i]);
        spoken += " and " + show(items.back());
        out.present = true;
        out.shown = show(value);
        out.expression = "the " + operation + " of " + spoken;
        out.fractional =
            render_exact(value.value).find('/') != std::string::npos;
        return out;
    } catch (const Undefined& refusal) {
        out.present = true;
        out.refusal = refusal.what();
        return out;
    }
}

}  // namespace uq
