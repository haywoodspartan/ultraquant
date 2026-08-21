// The enumerated-family question forms: superlatives and aggregates.
//
// Both stand on the same move - enumerate the WHOLE store for facts
// whose key ends in an attribute - and both inherit the same honesty
// contract: scope named, exclusions named by name, ties named rather
// than broken, and denials never counted in arithmetic, because a
// denial names no number and a denied 900 meters must not move a
// mean by a millimetre.
//
// A full scan is deliberate. A top-k sample could silently miss the
// winner, and a claim about a SET has to see the set.
#include "uq/forms.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>

#include "uq/calculate.hpp"
#include "uq/text.hpp"

namespace uq {
namespace {

std::string lower_strip(const std::string& text) {
    std::string low;
    for (char c : text)
        low.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    const std::string cut = "?!. ";
    const std::size_t begin = low.find_first_not_of(cut);
    if (begin == std::string::npos) return "";
    const std::size_t end = low.find_last_not_of(cut);
    return low.substr(begin, end - begin + 1);
}

bool starts_with(const std::string& text, const std::string& lead) {
    return text.size() >= lead.size()
        && text.compare(0, lead.size(), lead) == 0;
}

// Python's format(x, 'g') is C's %g, so the two tiers print a
// combined number the same way without either being asked to.
std::string g_format(double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%g", value);
    return buffer;
}

// CPython's sum() over floats is not a plain accumulation: since
// 3.12 it carries a Neumaier compensation term. The difference is
// real and visible - this family's average summed plainly is
// 1502.7749999999999 and summed with compensation is 1502.775,
// which %g prints as 250.462 and 250.463. Two tiers that disagree
// about a mean by one digit in the last place have not reproduced
// each other, and the fix is to sum the way the oracle sums rather
// than to widen the comparison.
double compensated_sum(const std::vector<double>& values) {
    double total = 0.0;
    double compensation = 0.0;
    for (double value : values) {
        const double moved = total + value;
        if (std::fabs(total) >= std::fabs(value))
            compensation += (total - moved) + value;
        else
            compensation += (value - moved) + total;
        total = moved;
    }
    return total + compensation;
}

std::string confidence_text(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    return buffer;
}

bool numeric_of(const std::string& value, double& out) {
    // The first number anywhere in the value, as _numeric finds it.
    std::size_t at = 0;
    while (at < value.size()) {
        const char c = value[at];
        if ((c >= '0' && c <= '9')
            || (c == '-' && at + 1 < value.size()
                && value[at + 1] >= '0' && value[at + 1] <= '9')) {
            std::size_t end = at + (c == '-' ? 1 : 0);
            while (end < value.size() && value[end] >= '0'
                   && value[end] <= '9') ++end;
            if (end + 1 < value.size() && value[end] == '.'
                && value[end + 1] >= '0' && value[end + 1] <= '9') {
                ++end;
                while (end < value.size() && value[end] >= '0'
                       && value[end] <= '9') ++end;
            }
            out = std::stod(value.substr(at, end - at));
            return true;
        }
        ++at;
    }
    return false;
}

// The unit word of a stored value, singular-folded.
std::string unit_of(const std::string& value) {
    std::string last;
    std::string current;
    for (char raw : value) {
        const char c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(raw)));
        if (c >= 'a' && c <= 'z') {
            current.push_back(c);
        } else if (!current.empty()) {
            last = current;
            current.clear();
        }
    }
    if (!current.empty()) last = current;
    return last.empty() ? "" : normalize_token(last);
}

}  // namespace

namespace {

struct Member {
    std::string key;
    Fact record;
    double number = 0.0;
    std::string unit;
};

struct Family {
    std::vector<Member> included;
    std::vector<std::pair<std::string, std::string>> excluded;
    int denied = 0;
};

// Enumerate the whole store for facts whose key ends in `attr`.
// A full scan by design: a top-k sample could silently miss a
// member, and a claim about a set must see the set.
Family attr_family(const Memory& memory, const std::string& attr) {
    Family out;
    for (const std::string& key : memory.fact_keys()) {
        std::vector<std::string> key_tokens;
        for (const std::string& token : raw_tokens(key))
            if (informative(token))
                key_tokens.push_back(normalize_token(token));
        if (key_tokens.empty() || key_tokens.back() != attr) continue;
        const Fact* record = memory.recall_fact(key);
        if (record == nullptr) continue;
        if (record->negated) { ++out.denied; continue; }
        double number = 0.0;
        if (!numeric_of(record->value, number)) {
            out.excluded.push_back({key, "no number"});
            continue;
        }
        Member member;
        member.key = key;
        member.record = *record;
        member.number = number;
        member.unit = unit_of(record->value);
        out.included.push_back(member);
    }
    return out;
}

const std::map<std::string, std::string>& superlative_words() {
    static const std::map<std::string, std::string> table = {
        {"tallest", "larger"}, {"longest", "larger"},
        {"biggest", "larger"}, {"highest", "larger"},
        {"largest", "larger"}, {"heaviest", "larger"},
        {"shortest", "smaller"}, {"smallest", "smaller"},
        {"lowest", "smaller"}, {"lightest", "smaller"},
    };
    return table;
}

const std::map<std::string, std::string>& implied_attrs() {
    static const std::map<std::string, std::string> table = {
        {"taller", "height"}, {"tallest", "height"},
        {"heavier", "weight"}, {"heaviest", "weight"},
        {"lighter", "weight"}, {"lightest", "weight"},
        {"longer", "length"}, {"longest", "length"},
    };
    return table;
}

}  // namespace

bool superlative_answer(const std::string& text, const Memory& memory,
                        std::string& said) {
    const std::string lowered = lower_strip(text);
    std::vector<std::string> tokens;
    for (const std::string& token : raw_tokens(lowered))
        if (informative(token)) tokens.push_back(normalize_token(token));
    std::vector<std::string> found;
    for (const std::string& token : tokens)
        if (superlative_words().count(token)) found.push_back(token);
    if (found.size() != 1
        || !(starts_with(lowered, "which ") || starts_with(lowered, "what ")))
        return false;
    const std::string word = found.front();
    const std::string direction = superlative_words().at(word);
    std::vector<std::string> attrs;
    for (const std::string& token : tokens)
        if (token != word) attrs.push_back(token);
    if (attrs.empty() && implied_attrs().count(word)) {
        // The word itself names the attribute where plain English
        // does; polysemous words imply nothing and refuse to guess.
        attrs.push_back(implied_attrs().at(word));
    }
    if (attrs.size() != 1) return false;
    const std::string attr = attrs.front();

    Family family = attr_family(memory, attr);
    if (family.included.empty()) {
        const std::string note = family.denied
            ? " (" + std::to_string(family.denied)
              + " denial(s) held, and a denial names no number)" : "";
        said = "I hold no " + attr + " facts to rank" + note + ".";
        return true;
    }

    // The result ranks in the family BASE, which is what the Python
    // tier compares in: multiply by the unit's factor, or leave a
    // family-less number alone. A unit outside the first member's
    // family is excluded BY NAME rather than silently dropped.
    const std::string first_unit = family.included.front().unit;
    std::string base_family;
    const bool has_family = unit_family(first_unit, base_family);
    std::vector<std::pair<Member, double>> comparable;
    for (const Member& member : family.included) {
        std::string member_family;
        const bool same = member.unit == first_unit
            || (has_family && unit_family(member.unit, member_family)
                && member_family == base_family);
        if (!same) {
            family.excluded.push_back(
                {member.key, "in " + (member.unit.empty()
                                      ? std::string("no unit") : member.unit)
                             + ", not comparable"});
            continue;
        }
        const double base = has_family
            ? member.number * unit_factor(member.unit) : member.number;
        comparable.push_back({member, base});
    }
    if (comparable.empty()) {
        said = "I hold " + attr + " facts but none share a comparable unit.";
        return true;
    }

    double best = comparable.front().second;
    for (const auto& item : comparable) {
        if (direction == "larger" ? item.second > best
                                  : item.second < best)
            best = item.second;
    }
    std::vector<Member> winners;
    for (const auto& item : comparable)
        if (item.second == best) winners.push_back(item.first);
    double confidence = comparable.front().first.record.confidence;
    for (const auto& item : comparable)
        confidence = std::min(confidence, item.first.record.confidence);
    const std::string scope = "of the " + std::to_string(comparable.size())
        + " " + attr + " facts I hold";
    std::string notes;
    if (!family.excluded.empty()) {
        std::string names;
        for (std::size_t i = 0; i < family.excluded.size(); ++i)
            names += (i ? "; " : "") + family.excluded[i].first + " ("
                + family.excluded[i].second + ")";
        notes += " Excluded: " + names + ".";
    }
    if (family.denied) {
        notes += " " + std::to_string(family.denied)
            + " denial(s) not counted - a denial names no number.";
    }
    if (winners.size() > 1) {
        std::string tied;
        for (std::size_t i = 0; i < winners.size(); ++i)
            tied += (i ? " and " : "") + winners[i].key;
        said = "Tied for " + word + " " + scope + ": " + tied + ", at "
            + winners.front().record.value + " each (confidence "
            + confidence_text(confidence) + ")." + notes;
    } else {
        said = "The " + winners.front().key + " is the " + word + " "
            + scope + ": " + winners.front().record.value
            + " (confidence " + confidence_text(confidence) + ")."
            + notes;
    }
    return true;
}

bool aggregate_answer(const std::string& text, const Memory& memory,
                      std::string& said) {
    const std::string lowered = lower_strip(text);
    const bool counting = starts_with(lowered, "how many ");
    std::vector<std::string> tokens;
    for (const std::string& token : raw_tokens(lowered))
        if (informative(token)) tokens.push_back(normalize_token(token));

    static const std::map<std::string, std::string> agg_words = {
        {"total", "sum"}, {"sum", "sum"},
        {"average", "mean"}, {"mean", "mean"},
    };
    static const std::set<std::string> noise = {
        "fact", "hold", "many", "how", "know",
    };
    std::vector<std::string> agg, rest;
    for (const std::string& token : tokens) {
        if (agg_words.count(token)) agg.push_back(token);
        else if (!noise.count(token)) rest.push_back(token);
    }
    std::string operation;
    if (counting) {
        if (rest.size() != 1 || !agg.empty()) return false;
        operation = "count";
    } else {
        if (agg.size() != 1 || rest.size() != 1
            || !starts_with(lowered, "what "))
            return false;
        operation = agg_words.at(agg.front());
    }
    const std::string attr = rest.front();

    Family family = attr_family(memory, attr);
    if (operation == "count") {
        // Counting tells the whole truth: positive beliefs counted,
        // denials reported beside them as what they are.
        const std::size_t held = family.included.size()
            + family.excluded.size();
        if (held == 0 && family.denied == 0) {
            said = "I hold no " + attr + " facts.";
        } else {
            const std::string note = family.denied
                ? ", plus " + std::to_string(family.denied) + " denial(s)"
                : "";
            said = "I hold " + std::to_string(held) + " " + attr
                + " fact(s)" + note + ".";
        }
        return true;
    }

    if (family.included.empty()) {
        const std::string note = family.denied
            ? " (" + std::to_string(family.denied)
              + " denial(s) held, and a denial names no number)" : "";
        said = "I hold no numeric " + attr + " facts" + note + ".";
        return true;
    }

    const std::string first_unit = family.included.front().unit;
    std::string base_family;
    const bool has_family = unit_family(first_unit, base_family);
    std::vector<Member> members;
    bool converted = false;
    for (const Member& member : family.included) {
        std::string member_family;
        if (member.unit == first_unit) {
            members.push_back(member);
        } else if (has_family && unit_family(member.unit, member_family)
                   && member_family == base_family) {
            members.push_back(member);
            converted = true;
        } else {
            family.excluded.push_back(
                {member.key, "in " + (member.unit.empty()
                                      ? std::string("no unit") : member.unit)
                             + ", not comparable"});
        }
    }
    if (members.empty()) {
        said = "I hold " + attr + " facts but none share a comparable unit.";
        return true;
    }

    double total = 0.0;
    std::string unit_text;
    if (has_family) {
        // The answer reads in the LARGEST unit present, which is the
        // rule the combine path answers by.
        std::string out_unit = members.front().unit;
        for (const Member& member : members)
            if (unit_factor(member.unit) > unit_factor(out_unit))
                out_unit = member.unit;
        std::vector<double> scaled;
        for (const Member& member : members)
            scaled.push_back(member.number * unit_factor(member.unit)
                             / unit_factor(out_unit));
        total = compensated_sum(scaled);
        unit_text = " " + out_unit + "s";
    } else {
        std::vector<double> plain;
        for (const Member& member : members)
            plain.push_back(member.number);
        total = compensated_sum(plain);
    }
    const double value = operation == "sum"
        ? total : total / static_cast<double>(members.size());
    double confidence = members.front().record.confidence;
    for (const Member& member : members)
        confidence = std::min(confidence, member.record.confidence);

    const std::string scope = "of the " + std::to_string(members.size())
        + " " + attr + " facts I hold";
    std::string notes;
    if (!family.excluded.empty()) {
        std::string names;
        for (std::size_t i = 0; i < family.excluded.size(); ++i)
            names += (i ? "; " : "") + family.excluded[i].first + " ("
                + family.excluded[i].second + ")";
        notes += " Excluded: " + names + ".";
    }
    if (family.denied) {
        notes += " " + std::to_string(family.denied)
            + " denial(s) not counted - a denial names no number.";
    }
    const std::string word = operation == "sum" ? "total" : "average";
    const std::string mark = converted ? " (units converted)" : "";
    said = "The " + word + " " + scope + " is " + g_format(value)
        + unit_text + mark + " (confidence " + confidence_text(confidence)
        + ")." + notes;
    return true;
}

}  // namespace uq
