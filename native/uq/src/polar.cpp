#include "uq/polar.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "uq/calculate.hpp"
#include "uq/inference.hpp"
#include "uq/text.hpp"

namespace uq {
namespace {

std::string lower_of(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text)
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    return out;
}

std::string confidence_text(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    return buffer;
}

// The first number anywhere in a value - _numeric's regex search.
bool numeric_of(const std::string& value, double& out) {
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

std::string shown_value(const Fact& fact) {
    return fact.negated ? "not " + fact.value : fact.value;
}

// A written quantity as a stored value would read it back.
std::string shown_quantity(const Quantity& quantity) {
    const std::string text = render_exact(quantity.value);
    return quantity.unit.empty() ? text : text + " " + quantity.unit + "s";
}

std::string join(const std::vector<std::string>& words, std::size_t from,
                 std::size_t to) {
    std::string out;
    for (std::size_t i = from; i < to && i < words.size(); ++i) {
        if (!out.empty()) out.push_back(' ');
        out += words[i];
    }
    return out;
}

bool is_article(const std::string& word) {
    return word == "a" || word == "an" || word == "the";
}

bool is_negator(const std::string& word) {
    return word == "not" || word == "never";
}

// The comparative vocabulary, copied from reason/inference.py's
// _COMBINE_WORDS. Only the two directions a comparison can point in
// are read here; "sum" and "difference" name combinations, not
// comparisons, and fall through to the value matrix below.
const std::map<std::string, std::string>& combine_words() {
    static const std::map<std::string, std::string> table = {
        {"taller", "larger"}, {"longer", "larger"}, {"bigger", "larger"},
        {"larger", "larger"}, {"higher", "larger"}, {"heavier", "larger"},
        {"shorter", "smaller"}, {"smaller", "smaller"},
        {"lower", "smaller"}, {"lighter", "smaller"},
        {"greater", "larger"}, {"more", "larger"},
        {"less", "smaller"}, {"fewer", "smaller"},
        {"sum", "sum"}, {"total", "sum"}, {"together", "sum"},
        {"combined", "sum"}, {"difference", "difference"},
        {"gap", "difference"},
    };
    return table;
}

// A comparative word names its own attribute, so a bare operand can
// try the appended key before refusing. Polysemous words (bigger,
// higher) imply nothing and are deliberately absent.
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

namespace {

// "Is A taller than B?" - the comparative and equality branch.
//
// This branch OWNS comparative questions. Before it existed in the
// Python tier, a comparison fell through to the value matrix below,
// where "300 meters" is never the string "taller than 200 meters" -
// so the answer was always No, in both directions, which is a coin
// that always lands the same way. Missing, negated, non-numeric and
// unit-incomparable operands refuse ALOUD rather than falling
// through, because a refusal is information and a No is a verdict.
bool polar_compare(const std::vector<std::string>& words,
                   const Memory& memory, std::string& said,
                   bool& unported) {
    std::size_t comp_at = 0;
    bool found = false;
    std::size_t skip = 2;
    std::string direction;

    if (words.size() >= 3) {
        for (std::size_t index = 1; index + 1 < words.size(); ++index) {
            auto it = combine_words().find(words[index]);
            if (it == combine_words().end()) continue;
            if ((it->second == "larger" || it->second == "smaller")
                && words[index + 1] == "than") {
                comp_at = index;
                direction = it->second;
                found = true;
                break;
            }
        }
    }
    if (!found && words.size() >= 3) {
        // Equality is the comparison people ask most and the one this
        // branch could not hear until it was written down. "=",
        // "equals" and "equal to" pivot the same way "taller than"
        // does, so both sides go through the same machinery.
        for (std::size_t index = 1; index + 1 < words.size(); ++index) {
            if (words[index] == "=" || words[index] == "equals") {
                comp_at = index; skip = 1; direction = "equal";
                found = true;
                break;
            }
            if (words[index] == "equal" && words[index + 1] == "to") {
                comp_at = index; skip = 2; direction = "equal";
                found = true;
                break;
            }
        }
    }
    if (!found) return false;

    auto side = [&words](std::size_t from, std::size_t to) {
        std::size_t start = from;
        if (start < to && start < words.size() && is_article(words[start]))
            ++start;
        return join(words, start, to);
    };
    const std::string left_key = side(0, comp_at);
    const std::string right_key = side(comp_at + skip, words.size());
    if (left_key.empty() || right_key.empty()) return false;

    struct Side {
        std::string key;
        std::string value;
        double confidence = 0.0;
        double number = 0.0;
        std::string unit;
        std::string trail;
        bool literal = false;
        bool expression = false;
    };
    Side sides[2];
    const std::string keys[2] = {left_key, right_key};

    for (int which = 0; which < 2; ++which) {
        std::string key = keys[which];
        const Fact* record = memory.recall_fact(key);
        Fact scratch;
        std::string trail;

        if (record == nullptr) {
            // The comparative word names its attribute, so a bare
            // operand tries the appended key before refusing.
            auto it = implied_attrs().find(words[comp_at]);
            if (it != implied_attrs().end()) {
                const std::string tail = " " + it->second;
                if (key.size() < tail.size()
                    || key.compare(key.size() - tail.size(), tail.size(),
                                   tail) != 0) {
                    key = key + tail;
                    record = memory.recall_fact(key);
                }
            }
        }
        if (record == nullptr) {
            // An operand the store does not hold may be DERIVED -
            // with the split discipline (no modifier-rescued
            // subjects) and the polarity line (a derived denial holds
            // no number to compare).
            const Inference attempt = infer("what is the " + key + "?",
                                            memory);
            if (attempt.present && attempt.has_conclusion
                && attempt.answer.find("as a modifier") == std::string::npos
                && attempt.answer.find("as modifiers") == std::string::npos) {
                if (attempt.negated) {
                    said = "I can't compare those: I can only derive a "
                           "denial for '" + key + "' (" + key + " is not "
                         + attempt.conclusion_value + ").";
                    return true;
                }
                scratch.value = attempt.conclusion_value;
                scratch.confidence = attempt.confidence;
                record = &scratch;
                for (std::size_t i = 0; i + 1 < attempt.premises.size(); ++i) {
                    if (!trail.empty()) trail += ", ";
                    trail += attempt.premises[i].second;
                }
            }
        }
        if (record == nullptr) {
            // The side may be a written quantity rather than a key.
            // "greater than 200 meters" names no belief and needs
            // none - the number is right there in the question, and
            // refusing to read it was why every comparison against a
            // literal fell through to a fabricated No.
            Quantity written;
            if (read_quantity(key, written)) {
                scratch.value = shown_quantity(written);
                scratch.confidence = 1.0;
                record = &scratch;
                sides[which].literal = true;
            } else {
                // Or an EXPRESSION. "is 3 * 4 greater than 10?" named
                // '3 * 4' as an unheld key, which is true and
                // useless: the question carries its own left-hand
                // side, and the reader that evaluates it is here.
                const MathResult worked = evaluate("what is " + key + "?");
                const bool answered = worked.present && worked.refusal.empty()
                    && (!worked.shown.empty() || worked.has_bounds);
                if (answered) {
                    scratch.value = worked.shown;
                    scratch.confidence = 1.0;
                    record = &scratch;
                    sides[which].literal = true;
                    sides[which].expression = true;
                } else if (key.find_first_of("+*/^") != std::string::npos) {
                    // An expression this tier cannot read: operands
                    // that name BELIEFS need the derived-operand
                    // reader, which is not ported. Flagged so the
                    // gate can count it rather than mistake it for a
                    // difference of opinion.
                    unported = true;
                }
            }
        }
        if (record == nullptr) {
            said = "I can't compare those: I hold nothing for '" + key
                 + "'.";
            return true;
        }
        if (record->negated) {
            said = "I can't compare those: I hold only a denial for '"
                 + key + "' (" + key + " is not " + record->value + ").";
            return true;
        }
        double number = 0.0;
        if (!numeric_of(record->value, number)) {
            said = "I can't compare those: '" + key + "' holds no number ("
                 + record->value + ").";
            return true;
        }
        sides[which].key = key;
        sides[which].value = record->value;
        sides[which].confidence = record->confidence;
        sides[which].number = number;
        sides[which].unit = unit_of(record->value);
        sides[which].trail = trail;
    }

    const Side& left = sides[0];
    const Side& right = sides[1];
    std::string converted;
    bool wins = false;
    bool equal = false;
    if (left.unit != right.unit
        || left.unit.empty() != right.unit.empty()) {
        std::string l_family;
        std::string r_family;
        const bool l_known = unit_family(left.unit, l_family);
        const bool r_known = unit_family(right.unit, r_family);
        if (left.unit.empty() || right.unit.empty() || !l_known || !r_known
            || l_family != r_family) {
            said = "I can't compare those: '" + left.key + "' is in "
                 + (left.unit.empty() ? std::string("no unit") : left.unit)
                 + " and '" + right.key + "' in "
                 + (right.unit.empty() ? std::string("no unit") : right.unit)
                 + ", and no definition connects them.";
            return true;
        }
        const double r_in_l = right.number * unit_factor(right.unit)
                            / unit_factor(left.unit);
        wins = direction == "larger" ? left.number > r_in_l
             : direction == "smaller" ? left.number < r_in_l
             : left.number == r_in_l;
        equal = left.number == r_in_l;
        converted = " (units converted)";
    } else {
        wins = direction == "larger" ? left.number > right.number
             : direction == "smaller" ? left.number < right.number
             : left.number == right.number;
        equal = left.number == right.number;
    }
    const double confidence = std::min(left.confidence, right.confidence);
    const std::string l_mark = left.trail.empty()
        ? std::string() : " (derived via " + left.trail + ")";
    const std::string r_mark = right.trail.empty()
        ? std::string() : " (derived via " + right.trail + ")";
    // A written quantity is its own name: "200 meters is 200 meters"
    // would be a strange thing to read back. An expression names
    // itself AND its value, so the reading stays checkable.
    auto piece = [](const Side& s) {
        if (s.expression) return s.key + " is " + s.value;
        return s.literal ? s.value : s.key + " is " + s.value;
    };
    const std::string left_piece = piece(left);
    const std::string right_piece = piece(right);
    const std::string both = left_piece + l_mark + ", " + right_piece
                           + r_mark;
    const std::string tail = " (confidence " + confidence_text(confidence)
                           + ").";
    if (direction == "equal") {
        // "Yes - 2 + 2 is 4." reads like every other polar answer;
        // "Yes - 4, 4" reads like a machine agreeing with itself.
        if (equal) {
            said = "Yes - " + left_piece + l_mark + converted + tail;
        } else {
            said = "No - " + left_piece + l_mark + ", not " + right.value
                 + converted + tail;
        }
    } else if (equal) {
        said = "No - they are equal: " + both + converted + tail;
    } else if (wins) {
        said = "Yes - " + both + converted + tail;
    } else {
        said = "No - " + both + converted + tail;
    }
    return true;
}

}  // namespace

namespace {

// "Is X steel or iron?" - the choice branch.
//
// The disjuncts are offered VALUES: the held one answers by name,
// "Neither" answers with the actual, a stored denial rules out its
// own value without electing another - ruling out is not picking -
// and an unheld subject falls through to the hedge, because absence
// never picks a side either.
bool polar_choice(const std::vector<std::string>& words,
                  const Memory& memory, std::string& said) {
    const Fact* fact = nullptr;
    std::string key;
    std::vector<std::string> claim_words;
    for (std::size_t size = words.size() - 1; size >= 1; --size) {
        bool has_or = false;
        for (std::size_t i = size; i < words.size(); ++i)
            if (words[i] == "or") { has_or = true; break; }
        if (!has_or) continue;
        const std::string candidate = join(words, 0, size);
        const Fact* record = memory.recall_fact(candidate);
        if (record != nullptr) {
            fact = record;
            key = candidate;
            claim_words.assign(words.begin() + static_cast<long>(size),
                               words.end());
            break;
        }
    }
    if (fact == nullptr) return false;

    std::vector<std::vector<std::string>> groups(1);
    for (const std::string& word : claim_words) {
        if (word == "or") groups.emplace_back();
        else if (!is_article(word)) groups.back().push_back(word);
    }
    std::vector<std::string> alternatives;
    for (const auto& group : groups) {
        if (group.empty()) continue;
        std::string joined;
        for (const std::string& word : group) {
            if (!joined.empty()) joined.push_back(' ');
            joined += word;
        }
        alternatives.push_back(joined);
    }
    if (alternatives.size() < 2) return false;

    const std::set<std::string> held = folded_tokens(fact->value);
    std::string matched;
    bool any = false;
    for (const std::string& alt : alternatives) {
        if (folded_tokens(alt) == held) { matched = alt; any = true; break; }
    }
    const std::string confidence = " (confidence "
        + confidence_text(fact->confidence) + ").";
    if (!fact->negated) {
        if (any) {
            // Speak the STORED value, not the parsed alternative:
            // article-stripping made "wren the younger" parse as
            // "wren younger", and the lead must say the belief as
            // held.
            std::string lead = fact->value;
            if (!lead.empty())
                lead[0] = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(lead[0])));
            said = lead + " - " + key + " is " + fact->value + confidence;
        } else {
            said = "Neither - " + key + " is " + fact->value + confidence;
        }
    } else if (any) {
        // Ruling out is not picking: the other disjunct stays
        // unelected, because a denial of one value affirms nothing
        // about another.
        said = "Not " + fact->value + ", at least - I hold only that "
             + key + " is not " + fact->value + confidence;
    } else {
        said = "I don't know - I hold only that " + key + " is not "
             + fact->value + confidence;
    }
    return true;
}

}  // namespace

namespace {

// Answer a polar question by DERIVING its subject.
//
// "Is the dome city climate temperate?" with only "dome city is
// york" and "york climate is not temperate" held: the chain
// machinery answers "what is the dome city climate?" and the derived
// value meets the claim under the same matrix direct facts use -
// polarity included, absence still never no, and the reply marked
// derived with its trail so the verdict can be vetoed premise by
// premise.
bool polar_derive(const std::vector<std::string>& words, std::size_t size,
                  const Memory& memory, std::string& said) {
    if (size > 0 && is_negator(words[size - 1])) {
        // The split landed mid-claim: "... climate not" / "temperate"
        // absorbed the polarity word into the subject and answered No
        // where the claim agreed. The negator belongs to the claim;
        // let the next size put it there.
        return false;
    }
    bool claim_negated = false;
    std::string claimed;
    for (std::size_t i = size; i < words.size(); ++i) {
        if (is_negator(words[i])) { claim_negated = true; continue; }
        if (is_article(words[i])) continue;
        if (!claimed.empty()) claimed.push_back(' ');
        claimed += words[i];
    }
    if (claimed.empty()) return false;

    const std::string subject = join(words, 0, size);
    const Inference derived = infer("what is the " + subject + "?", memory);
    if (!derived.present) return false;
    if (derived.answer.find("as a modifier") != std::string::npos
        || derived.answer.find("as modifiers") != std::string::npos) {
        // The derivation only converged by reading part of THIS
        // subject away - which means the split was wrong, and the
        // dropped word belongs to the claim. Let a shorter subject
        // try.
        return false;
    }
    const std::set<std::string> want = folded_tokens(claimed);
    const std::set<std::string> got = folded_tokens(
        derived.has_conclusion ? derived.conclusion_value : std::string());
    const bool matches = want == got;
    const std::string suffix = " (derived, confidence "
        + confidence_text(derived.confidence) + ").";
    if (!derived.negated) {
        const std::string verdict = (matches == !claim_negated)
            ? "Yes" : "No";
        said = verdict + " - " + derived.answer + suffix;
    } else if (matches) {
        said = std::string(claim_negated ? "Yes" : "No") + " - "
             + derived.answer + suffix;
    } else {
        // A derived negation of one value says nothing about another
        // - the same line the stored case keeps.
        said = "I don't know - I can only derive that " + derived.answer
             + suffix;
    }
    return true;
}

}  // namespace

bool polar_answer_gapped(const std::string& text, const Memory& memory,
                         std::string& said, bool& unported) {
    std::string lowered = lower_of(text);
    const std::string cut = "?!. ";
    std::size_t begin = lowered.find_first_not_of(cut);
    if (begin == std::string::npos) return false;
    std::size_t end = lowered.find_last_not_of(cut);
    lowered = lowered.substr(begin, end - begin + 1);

    std::string rest;
    bool opened = false;
    for (const std::string& lead : {"is ", "are ", "was ", "were "}) {
        if (lowered.size() > lead.size()
            && lowered.compare(0, lead.size(), lead) == 0) {
            rest = lowered.substr(lead.size());
            std::size_t at = rest.find_first_not_of(' ');
            rest = at == std::string::npos ? std::string() : rest.substr(at);
            std::size_t back = rest.find_last_not_of(' ');
            if (back != std::string::npos) rest = rest.substr(0, back + 1);
            opened = true;
            break;
        }
    }
    if (!opened) return false;
    for (const std::string& article : {"the ", "a ", "an "}) {
        if (rest.size() > article.size()
            && rest.compare(0, article.size(), article) == 0)
            rest = rest.substr(article.size());
    }

    std::vector<std::string> words;
    {
        std::string current;
        for (char c : rest) {
            if (c == ' ') {
                if (!current.empty()) { words.push_back(current);
                                        current.clear(); }
            } else {
                current.push_back(c);
            }
        }
        if (!current.empty()) words.push_back(current);
    }
    if (rest.find('=') != std::string::npos) {
        // "2+2=4" is a single token to a space split; the equals sign
        // is an operator wherever it is written, so it becomes its
        // own token before anything counts words.
        std::vector<std::string> split;
        for (const std::string& word : words) {
            std::string current;
            for (char c : word) {
                if (c == '=') {
                    if (!current.empty()) { split.push_back(current);
                                            current.clear(); }
                    split.push_back("=");
                } else {
                    current.push_back(c);
                }
            }
            if (!current.empty()) split.push_back(current);
        }
        words.swap(split);
    }
    if (words.size() < 2) return false;

    if (polar_compare(words, memory, said, unported)) return true;
    bool has_or = false;
    for (const std::string& word : words)
        if (word == "or") { has_or = true; break; }
    if (has_or && polar_choice(words, memory, said)) return true;

    // One ladder, longest subject first, where "subject" means a
    // stored key OR a derivable one. "dome city climate temperate"
    // must try deriving "dome city climate" BEFORE the stored "dome
    // city" claims the split - the first cut answered about the city
    // when the question asked about the climate.
    const Fact* fact = nullptr;
    std::string key;
    std::size_t at = 0;
    for (std::size_t size = words.size() - 1; size >= 1; --size) {
        const std::string candidate = join(words, 0, size);
        const Fact* record = memory.recall_fact(candidate);
        if (record != nullptr) {
            fact = record;
            key = candidate;
            at = size;
            break;
        }
        if (words.size() - size <= 2 && size >= 2
            && polar_derive(words, size, memory, said))
            return true;
    }
    if (fact == nullptr) return false;

    bool claim_negated = false;
    bool has_than = false;
    std::string claimed;
    std::size_t than_at = 0;
    for (std::size_t i = at; i < words.size(); ++i) {
        if (words[i] == "than" && !has_than) { has_than = true; than_at = i; }
        if (is_negator(words[i])) { claim_negated = true; continue; }
        if (is_article(words[i])) continue;
        if (!claimed.empty()) claimed.push_back(' ');
        claimed += words[i];
    }
    if (claimed.empty()) return false;

    if (has_than) {
        // This is a COMPARISON, and the branch above owns comparisons.
        // Reaching here means that branch declined - the word is one
        // this system cannot compare by - and a comparison is not
        // answered by checking whether the stored value happens to
        // spell the comparison out. Before this line, "is the tower
        // height greater than 200 meters?" answered "No - tower
        // height is 300 meters, not greater than 200 meters", and so
        // did every other comparison in both directions: a coin that
        // always said No.
        const std::string word = than_at > at ? words[than_at - 1]
                                              : std::string("that");
        said = "I can't tell: I don't know how to compare by '" + word
             + "'. I hold that " + key + " is " + shown_value(*fact) + ".";
        return true;
    }

    const bool matches = folded_tokens(claimed) == folded_tokens(fact->value);
    const std::string confidence = "(confidence "
        + confidence_text(fact->confidence) + ")";
    if (!fact->negated) {
        if (matches) {
            said = std::string(claim_negated ? "No" : "Yes") + " - " + key
                 + " is " + fact->value + " " + confidence + ".";
        } else if (claim_negated) {
            said = "Yes - " + key + " is " + fact->value + ", not " + claimed
                 + " " + confidence + ".";
        } else {
            said = "No - " + key + " is " + fact->value + ", not " + claimed
                 + " " + confidence + ".";
        }
    } else if (matches) {
        said = std::string(claim_negated ? "Yes" : "No")
             + " - believed not " + fact->value + " " + confidence + ".";
    } else {
        // A negation of one value says nothing about another: "not
        // steel" cannot answer "is it iron?".
        said = "I don't know - I hold only that " + key + " is not "
             + fact->value + " " + confidence + ".";
    }
    return true;
}

bool polar_answer(const std::string& text, const Memory& memory,
                  std::string& said) {
    bool unported = false;
    return polar_answer_gapped(text, memory, said, unported);
}

}  // namespace uq
