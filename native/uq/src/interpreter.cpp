#include "uq/interpreter.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <set>
#include <sstream>

#include "uq/calculate.hpp"
#include "uq/inference.hpp"

namespace uq {
namespace {

const char* kInterrogativeLeads[] = {
    "what ", "who ", "where ", "when ", "which ", "why ",
    "how ", "is ", "are ", "does ", "do ", "can ",
};

std::string lower(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text)
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    return out;
}

std::string trim(const std::string& text) {
    const std::size_t begin = text.find_first_not_of(" \t");
    if (begin == std::string::npos) return "";
    const std::size_t end = text.find_last_not_of(" \t");
    return text.substr(begin, end - begin + 1);
}

std::string rstrip(const std::string& text, const std::string& cut) {
    std::size_t end = text.size();
    while (end > 0 && cut.find(text[end - 1]) != std::string::npos) --end;
    return text.substr(0, end);
}

bool starts_with(const std::string& text, const std::string& lead) {
    return text.size() >= lead.size()
        && text.compare(0, lead.size(), lead) == 0;
}

bool ends_with(const std::string& text, const std::string& tail) {
    return text.size() >= tail.size()
        && text.compare(text.size() - tail.size(), tail.size(), tail) == 0;
}

bool has_interrogative_lead(const std::string& lowered) {
    for (const char* lead : kInterrogativeLeads)
        if (starts_with(lowered, lead)) return true;
    return false;
}

std::string confidence_text(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

std::string shown_value(const Fact& fact) {
    return fact.negated ? "not " + fact.value : fact.value;
}

// "not steel" and "never steel" are belief-of-absence: the bare
// value is stored and the polarity travels as a flag, so a denial
// can never be bridged as an assertion.
void split_polarity(const std::string& value, std::string& bare,
                    bool& negated) {
    const std::string lowered = lower(value);
    for (const std::string& lead : {std::string("not "),
                                    std::string("never ")}) {
        if (starts_with(lowered, lead)) {
            const std::string rest = trim(value.substr(lead.size()));
            if (!rest.empty()) {
                bare = rest;
                negated = true;
                return;
            }
        }
    }
    bare = value;
    negated = false;
}

bool parse_statement(const std::string& text, std::string& key,
                     std::string& value) {
    std::string cleaned = rstrip(trim(text), ".");
    std::string lowered = lower(cleaned);
    for (const std::string& lead : {std::string("remember that "),
                                    std::string("remember, "),
                                    std::string("remember: "),
                                    std::string("remember ")}) {
        if (starts_with(lowered, lead)) {
            cleaned = cleaned.substr(lead.size());
            lowered = lower(cleaned);
            break;
        }
    }
    const std::size_t equals = cleaned.find('=');
    if (equals != std::string::npos) {
        key = lower(trim(cleaned.substr(0, equals)));
        value = trim(cleaned.substr(equals + 1));
        return true;
    }
    const std::size_t at = lowered.find(" is ");
    if (at != std::string::npos && at > 0) {
        key = lower(trim(cleaned.substr(0, at)));
        value = trim(cleaned.substr(at + 4));
        for (const std::string& article : {std::string("the "),
                                           std::string("a "),
                                           std::string("an ")}) {
            if (starts_with(key, article)) {
                key = key.substr(article.size());
                break;
            }
        }
        return !key.empty() && !value.empty();
    }
    return false;
}

}  // namespace

namespace {

// The stopword list the router uses, because §11.29's coverage rule
// asks whether the question's INFORMATIVE tokens are covered - and a
// key that shares "the" with a question has been told nothing.
bool informative(const std::string& token) {
    // The interpreter's own stopword list, copied rather than
    // reinvented: §11.29's coverage rule asks whether a question's
    // INFORMATIVE tokens are covered by a key, so a native tier with
    // a different list would assert where Python hedges. It is also
    // a length test - nothing of three characters or fewer is
    // informative, which is why "is" and "of" would be excluded even
    // if the list forgot them.
    static const std::set<std::string> stop = {
        "a", "about", "all", "an", "and", "any", "are", "as", "at",
        "be", "been", "being", "but", "by", "can", "could", "did",
        "do", "does", "done", "each", "every", "for", "from", "get",
        "give", "got", "had", "has", "have", "here", "how", "i",
        "if", "in", "into", "is", "it", "it's", "its", "just",
        "know", "let", "look", "make", "many", "may", "me", "might",
        "mine", "more", "most", "much", "must", "my", "need", "no",
        "not", "of", "on", "or", "other", "our", "ours", "same",
        "see", "should", "show", "so", "some", "take", "tell",
        "than", "that", "the", "their", "them", "then", "there",
        "these", "they", "think", "this", "those", "to", "us", "use",
        "used", "using", "very", "want", "was", "we", "were", "what",
        "when", "where", "which", "who", "whom", "why", "will",
        "with", "would", "yes", "you", "your", "yours"
    };
    return token.size() > 2 && stop.find(token) == stop.end();
}

std::set<std::string> informative_tokens(const std::string& text) {
    std::set<std::string> out;
    for (const std::string& token : tokens_of(text))
        if (informative(token)) out.insert(token);
    return out;
}

// Fact keys worth trying for this input, most specific first - the
// same ladder, including the article-stripped variants, because keys
// are stored with their articles removed.
std::vector<std::string> candidate_keys(const std::string& text) {
    const std::string lowered = rstrip(trim(lower(text)), "?.!");
    std::vector<std::string> keys;
    for (const std::string& lead : {std::string("what is "),
                                    std::string("who is "),
                                    std::string("where is "),
                                    std::string("what are "),
                                    std::string("remember that ")}) {
        if (starts_with(lowered, lead))
            keys.push_back(trim(lowered.substr(lead.size())));
    }
    const std::size_t at = lowered.find(" is ");
    if (at != std::string::npos) keys.push_back(trim(lowered.substr(0, at)));
    keys.push_back(lowered);
    const std::vector<std::string> so_far = keys;
    for (const std::string& key : so_far) {
        for (const std::string& article : {std::string("the "),
                                           std::string("a "),
                                           std::string("an ")}) {
            if (starts_with(key, article))
                keys.push_back(key.substr(article.size()));
        }
    }
    // Progressively shorter n-grams, longest first, so the most
    // specific stored key wins when several match.
    const std::vector<std::string> tokens = tokens_of(text);
    for (int size = 5; size >= 1; --size) {
        if (static_cast<int>(tokens.size()) < size) continue;
        for (std::size_t i = 0; i + size <= tokens.size(); ++i) {
            std::string gram;
            for (int j = 0; j < size; ++j) {
                if (j) gram += " ";
                gram += tokens[i + j];
            }
            keys.push_back(gram);
        }
    }
    std::vector<std::string> ordered;
    std::set<std::string> seen;
    for (const std::string& key : keys) {
        if (key.empty() || seen.count(key)) continue;
        seen.insert(key);
        ordered.push_back(key);
        if (ordered.size() >= 24) break;
    }
    return ordered;
}

}  // namespace

Turn Session::run(const std::string& text) {
    const std::string lowered = lower(text);
    // An interrogative lead beats the " is " heuristic: "how tall is
    // the tower" contains " is " and would otherwise be read as a
    // statement, which both refuses to answer it and stores junk.
    const bool statement_shaped =
        starts_with(lowered, "remember")
        || (lowered.find(" is ") != std::string::npos
            && !ends_with(trim(text), "?")
            && !has_interrogative_lead(lowered));
    if (statement_shaped) return learn_statement(text);
    if (ends_with(trim(text), "?") || has_interrogative_lead(lowered))
        return answer_question(text);
    // An expression is self-identifying - nothing but numbers,
    // operators and parentheses - so admitting it steals nothing: a
    // statement carries words, and words make this false.
    const MathResult expression = evaluate(text);
    if (expression.present) return answer_question(text);
    Turn turn;
    turn.intent = "chat";
    turn.response = "I have nothing on that yet. ':help' lists what I can do.";
    return turn;
}

Turn Session::learn_statement(const std::string& text) {
    Turn turn;
    turn.intent = "fact_statement";
    std::string key, raw_value;
    if (!parse_statement(text, key, raw_value)) {
        turn.intent = "chat";
        turn.response =
            "I have nothing on that yet. ':help' lists what I can do.";
        return turn;
    }
    std::string value;
    bool negated = false;
    split_polarity(raw_value, value, negated);
    const StoreOutcome result =
        memory_.remember_fact(key, value, 0.6, negated);
    turn.response = "Noted: " + key + " is " + raw_value + ".";

    // §11.63: a MORE SPECIFIC subject stores a new fact, and silence
    // about the held base invites the reader to assume the two are
    // one. The adjacency is spoken, never treated as a conflict.
    if (result.outcome == "new") {
        const std::vector<std::string> parts = tokens_of(key);
        std::vector<std::string> words;
        std::istringstream in(key);
        std::string word;
        while (in >> word) words.push_back(word);
        if (words.size() >= 3) {
            std::string base;
            for (std::size_t i = 1; i < words.size(); ++i)
                base += (i > 1 ? " " : "") + words[i];
            const Fact* held = memory_.recall_fact(base);
            if (held != nullptr) {
                turn.response += " I separately hold: " + base + " is "
                    + shown_value(*held) + " (confidence "
                    + confidence_text(held->confidence) + ").";
            }
        }
    }
    if (result.outcome == "revised") {
        // The old belief is named so it can be defended, and the
        // retractions are counted so the cost is visible.
        std::string notice = "That revises what I held: " + key + " was "
            + result.was + ".";
        if (!result.retracted.empty()) {
            std::string names;
            for (std::size_t i = 0; i < result.retracted.size(); ++i)
                names += (i ? ", " : "") + ("'" + result.retracted[i] + "'");
            notice += " " + std::to_string(result.retracted.size())
                + " derived fact(s) rested on it and were retracted: "
                + names + ".";
        }
        turn.response += " " + notice;
    }
    return turn;
}

Turn Session::answer_question(const std::string& text) {
    Turn turn;
    turn.intent = "question";

    const MathResult expression = evaluate(text);
    if (expression.present) {
        if (!expression.refusal.empty()) {
            turn.response = "I can't compute that: " + expression.refusal
                + ".";
            return turn;
        }
        if (expression.has_bounds) {
            turn.response = expression.expression + " is between "
                + expression.low + " and " + expression.high
                + " - that value is irrational, so those are proved "
                  "bounds, not the number.";
            return turn;
        }
        if (expression.rounded_to >= 0) {
            const std::string where = expression.rounded_to == 0
                ? std::string("the nearest whole number")
                : std::to_string(expression.rounded_to) + " decimal place"
                  + (expression.rounded_to == 1 ? "" : "s");
            if (expression.was_rounded) {
                const std::string exact =
                    expression.exact_shown == "irrational"
                        ? std::string("irrational") : expression.exact_shown;
                turn.response = expression.expression + " = "
                    + expression.shown + " - rounded to " + where
                    + "; the exact value is " + exact + ".";
            } else {
                turn.response = expression.expression + " = "
                    + expression.shown + " - exact at " + where
                    + ", so nothing was rounded.";
            }
            return turn;
        }
        const std::string exact = expression.fractional
            ? " (exact - that value has no decimal form)" : "";
        turn.response = expression.expression + " = " + expression.shown
            + exact + " - computed, not stored.";
        return turn;
    }

    const MathResult listed = read_list(text);
    if (listed.present) {
        if (!listed.refusal.empty()) {
            turn.response = "I can't compute that: " + listed.refusal + ".";
            return turn;
        }
        const std::string exact = listed.fractional
            ? " (exact - that value has no decimal form)" : "";
        turn.response = listed.expression + " = " + listed.shown + exact
            + " - computed, not stored.";
        return turn;
    }

    // The exact branch, under §11.29's coverage rule: a hit here is
    // not necessarily the whole question, so assert only when the
    // key covers everything the question asked about.
    const std::set<std::string> asked = informative_tokens(text);
    for (const std::string& key : candidate_keys(text)) {
        const Fact* fact = memory_.recall_fact(key);
        if (fact == nullptr) continue;
        const std::set<std::string> held = informative_tokens(key);
        if (std::includes(held.begin(), held.end(),
                          asked.begin(), asked.end())) {
            turn.response = key + " is " + shown_value(*fact)
                + " (confidence " + confidence_text(fact->confidence) + ").";
            return turn;
        }
        break;   // a sub-key hit that does not cover falls through
    }

    // Derivation runs BEFORE the loose keyword fallback: its coverage
    // rules are strict, so when a chain exists it answers the whole
    // question - where the keyword fallback would have answered
    // whichever single premise ranked first and stopped.
    const Inference derived = infer(text, memory_);
    if (derived.present) {
        turn.response = derived.describe();
        return turn;
    }

    // A failed convergence knows which premise was missing, and a
    // grounded gap becomes a question worth asking instead of a dead
    // end. Whatever the hedges below settle on carries the hint.
    std::string hint;
    const MissingPremise gap = missing_premise(text, memory_);
    if (gap.present) {
        hint = " If I knew the " + gap.premise_key
            + ", I could work this out - ':learn' will ask.";
    }

    // Keyword overlap, then the same coverage rule again: nearest-held
    // is worth SAYING, and it is not worth asserting as identity.
    std::vector<std::string> candidates = memory_.find_facts(text, 3);
    std::string folded;
    for (const std::string& token : asked)
        folded += (folded.empty() ? "" : " ") + token;
    for (const std::string& key : memory_.find_facts(folded, 3)) {
        if (std::find(candidates.begin(), candidates.end(), key)
            == candidates.end())
            candidates.push_back(key);
    }
    for (const std::string& key : candidates) {
        const std::set<std::string> held = informative_tokens(key);
        bool shares = false;
        for (const std::string& token : held)
            if (asked.count(token)) { shares = true; break; }
        if (!shares) continue;
        const Fact* fact = memory_.recall_fact(key);
        if (fact == nullptr) continue;
        if (std::includes(held.begin(), held.end(),
                          asked.begin(), asked.end())) {
            turn.response = key + " is " + shown_value(*fact)
                + " (confidence " + confidence_text(fact->confidence) + ").";
            return turn;
        }
        turn.response = "I don't hold that exactly. Nearest I hold: " + key
            + " is " + shown_value(*fact) + " (confidence "
            + confidence_text(fact->confidence) + ")." + hint;
        return turn;
    }
    turn.response =
        "I don't hold anything on that yet. Tell me directly ('X is Y'), "
        "or give me a URL and I'll stash what it claims for analysis."
        + hint;
    return turn;
}

}  // namespace uq
