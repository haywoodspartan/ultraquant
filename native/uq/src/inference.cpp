#include "uq/inference.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "uq/calculate.hpp"

namespace uq {
namespace {

const double kDecay = 0.5;
const double kFloor = 0.05;
const int kMaxHops = 4;
const int kMaxModifiers = 3;

bool informative(const std::string& token) {
    // The interpreter's own list, copied rather than reinvented: the
    // spread's coverage test asks whether a question's INFORMATIVE
    // tokens are covered, so a different list is a different answer.
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

std::vector<std::string> raw_lower_tokens(const std::string& text) {
    std::vector<std::string> out;
    std::string current;
    for (char raw : text) {
        const char c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(raw)));
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            current.push_back(c);
        } else if (!current.empty()) {
            out.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

// The question's informative tokens, folded. A std::set is sorted,
// which is exactly what the Python side asks for wherever the order
// of a token set is observable (it always calls sorted()).
std::set<std::string> fold(const std::string& text) {
    std::set<std::string> out;
    for (const std::string& token : raw_lower_tokens(text))
        if (informative(token)) out.insert(normalize_token(token));
    return out;
}

// The same tokens in QUESTION order, deduplicated - order is
// evidence at density, so this is not a convenience.
std::vector<std::string> ordered_fold(const std::string& text) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const std::string& token : raw_lower_tokens(text)) {
        const std::string folded = normalize_token(token);
        if (seen.count(folded) || !informative(token)) continue;
        seen.insert(folded);
        out.push_back(folded);
    }
    return out;
}

// Consecutive trigrams then bigrams: a subject is a PHRASE, and
// probing chunks ranks its facts above every single-word sharer.
std::vector<std::string> probe_phrases(
    const std::vector<std::string>& tokens) {
    std::vector<std::string> out;
    for (int size : {3, 2}) {
        if (static_cast<int>(tokens.size()) < size) continue;
        for (std::size_t start = 0; start + size <= tokens.size();
             ++start) {
            std::string phrase;
            for (int i = 0; i < size; ++i)
                phrase += (i ? " " : "") + tokens[start + i];
            out.push_back(phrase);
        }
    }
    return out;
}

// An insertion-ordered key->record view: the paging boundary, and
// the order the spread iterates.
struct FactSet {
    std::vector<std::string> order;
    std::map<std::string, Fact> data;

    bool has(const std::string& key) const { return data.count(key) != 0; }
    void add(const std::string& key, const Fact& fact) {
        if (has(key)) return;
        order.push_back(key);
        data[key] = fact;
    }
};

FactSet reachable_facts(const Memory& memory,
                        const std::vector<std::string>& probes,
                        std::size_t top_k = 8) {
    FactSet found;
    for (const std::string& probe : probes) {
        for (const std::string& key : memory.find_facts(probe, top_k)) {
            if (found.has(key)) continue;
            const Fact* record = memory.recall_fact(key);
            if (record != nullptr) found.add(key, *record);
        }
    }
    return found;
}

}  // namespace

namespace {

// One activated fact during a spread. Bridged evidence lives in
// origins, one entry per lineage, never pooled - the single-origin
// rule that keeps a crowded library from assembling puns out of
// unrelated subjects.
struct Lineage {
    std::map<std::string, double> sources;
    std::vector<std::pair<std::string, std::string>> path;
};

struct Node {
    std::string key;
    Fact record;
    std::map<std::string, double> sources;         // direct contact
    std::vector<std::string> origin_order;         // insertion order
    std::map<std::string, Lineage> origins;
};

struct NodeSet {
    std::vector<std::string> order;
    std::map<std::string, Node> data;

    Node* find(const std::string& key) {
        const auto at = data.find(key);
        return at == data.end() ? nullptr : &at->second;
    }
    Node& add(const Node& node) {
        if (!data.count(node.key)) order.push_back(node.key);
        data[node.key] = node;
        return data[node.key];
    }
};

double sum_of(const std::map<std::string, double>& values) {
    double total = 0.0;
    for (const auto& entry : values) total += entry.second;
    return total;
}

std::set<std::string> bigrams_of(const std::vector<std::string>& tokens) {
    std::set<std::string> out;
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
        out.insert(tokens[i] + " " + tokens[i + 1]);
    return out;
}

std::string join_with(const std::vector<std::string>& parts,
                      const std::string& sep) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i)
        out += (i ? sep : "") + parts[i];
    return out;
}

std::string confidence_text(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

}  // namespace

std::string Inference::describe() const {
    std::vector<std::string> parts;
    for (const auto& premise : premises)
        parts.push_back(premise.first + " is " + premise.second);
    return answer + " - inferred, not stored: " + join_with(parts, "; ")
        + " (confidence " + confidence_text(confidence) + ")";
}

namespace {

Inference spread(const std::set<std::string>& question_tokens,
                 const Memory& memory,
                 const std::vector<std::string>& ordered) {
    Inference nothing;
    std::vector<std::string> probes;
    probes.push_back(join_with(
        std::vector<std::string>(question_tokens.begin(),
                                 question_tokens.end()), " "));
    for (const std::string& phrase : probe_phrases(ordered))
        probes.push_back(phrase);
    for (const std::string& token : question_tokens)
        probes.push_back(token);

    const FactSet facts = reachable_facts(memory, probes);
    if (facts.order.empty()) return nothing;

    NodeSet nodes;
    for (const std::string& key : facts.order) {
        const Fact& record = facts.data.at(key);
        // A negation is inhibitory knowledge: it says what may NOT be
        // believed, so it neither seeds nor relays. Letting it excite
        // the network bridged a denial into an assertion.
        if (record.negated) continue;
        const std::set<std::string> key_tokens = fold(key);
        Node node;
        node.key = key;
        node.record = record;
        for (const std::string& token : key_tokens)
            if (question_tokens.count(token)) node.sources[token] = 1.0;
        if (node.sources.empty()) continue;
        nodes.add(node);
    }

    struct Pending {
        std::string key;
        Fact record;
        std::string origin;
        std::map<std::string, double> arriving;
        std::vector<std::pair<std::string, std::string>> path;
    };

    bool have_frontier = false;
    std::set<std::string> frontier;
    std::map<std::string, FactSet> probe_memo;
    for (int hop = 0; hop < kMaxHops; ++hop) {
        bool grown = false;
        std::vector<Pending> pending;
        std::set<std::string> changed;
        const std::vector<std::string> walking = nodes.order;
        for (const std::string& node_key : walking) {
            Node& node = *nodes.find(node_key);
            // Only nodes that gained or strengthened an origin last
            // round have anything NEW to relay; a node's own direct
            // sources never change after seeding.
            if (have_frontier && !frontier.count(node.key)) continue;
            if (node.record.negated) continue;   // reached, never relayed

            std::vector<std::pair<std::string, Lineage>> lineages;
            if (!node.sources.empty()) {
                Lineage own;
                own.sources = node.sources;
                lineages.push_back({node.key, own});
            }
            for (const std::string& origin : node.origin_order) {
                Lineage merged = node.origins.at(origin);
                for (const auto& entry : node.sources) {
                    auto at = merged.sources.find(entry.first);
                    if (at == merged.sources.end()
                        || at->second < entry.second)
                        merged.sources[entry.first] = entry.second;
                }
                lineages.push_back({origin, merged});
            }
            const std::string value = node.record.value;
            const std::set<std::string> bridge_tokens = fold(value);
            if (bridge_tokens.empty()) continue;

            // Addressed bridge probes: a bare value is also a prefix
            // word in a dense store, and its ranked buckets drown the
            // keys that matter. Pairing it with each question token
            // addresses the target's own bucket directly.
            std::vector<std::string> bridge_probes{value};
            for (const std::string& token : question_tokens)
                bridge_probes.push_back(value + " " + token);
            if (!probe_memo.count(value))
                probe_memo[value] = reachable_facts(memory, bridge_probes);
            const FactSet& targets = probe_memo[value];

            for (const std::string& t_key : targets.order) {
                const Fact& t_record = targets.data.at(t_key);
                if (t_key == node.key) continue;
                const std::set<std::string> t_tokens = fold(t_key);
                // Whole-value bridging, both directions: every token
                // of a chain fact is accounted for - by the incoming
                // value, by the question, or by ONE terminal relation
                // slot - or the hop refuses. An unaccounted token
                // INSIDE the subject is a qualifier being dropped, and
                // dropping qualifiers is how one entity substitutes
                // for another.
                if (!std::includes(t_tokens.begin(), t_tokens.end(),
                                   bridge_tokens.begin(),
                                   bridge_tokens.end()))
                    continue;
                std::set<std::string> extra;
                for (const std::string& token : t_tokens) {
                    if (!bridge_tokens.count(token)
                        && !question_tokens.count(token))
                        extra.insert(token);
                }
                if (!extra.empty()) {
                    const std::vector<std::string> key_order =
                        ordered_fold(t_key);
                    if (extra.size() > 1 || key_order.empty()
                        || !extra.count(key_order.back()))
                        continue;
                }
                for (const auto& lineage : lineages) {
                    bool loops = false;
                    for (const auto& step : lineage.second.path)
                        if (step.first == t_key) { loops = true; break; }
                    if (loops) continue;
                    Pending item;
                    item.key = t_key;
                    item.record = t_record;
                    item.origin = lineage.first;
                    item.path = lineage.second.path;
                    item.path.push_back({node.key, node.record.value});
                    for (const auto& entry : lineage.second.sources) {
                        const double decayed = entry.second * kDecay;
                        if (decayed >= kFloor)
                            item.arriving[entry.first] = decayed;
                    }
                    if (item.arriving.empty()) continue;
                    pending.push_back(item);
                }
            }
        }
        for (const Pending& item : pending) {
            Node* target = nodes.find(item.key);
            if (target == nullptr) {
                Node fresh;
                fresh.key = item.key;
                fresh.record = item.record;
                for (const std::string& token : fold(item.key))
                    if (question_tokens.count(token))
                        fresh.sources[token] = 1.0;
                target = &nodes.add(fresh);
            }
            const auto at = target->origins.find(item.origin);
            if (at == target->origins.end()
                || sum_of(item.arriving) > sum_of(at->second.sources)) {
                if (at == target->origins.end())
                    target->origin_order.push_back(item.origin);
                Lineage lineage;
                lineage.sources = item.arriving;
                lineage.path = item.path;
                target->origins[item.origin] = lineage;
                changed.insert(item.key);
                grown = true;
            }
        }
        frontier = changed;
        have_frontier = true;
        if (!grown) break;
    }
    // Convergence: the target's DIRECT contact plus ONE origin's
    // bridged contribution must cover the whole question, and the
    // origin's path must be non-empty - all-direct coverage is plain
    // recall's turn. Order is EVIDENCE at density: "river wall barn"
    // and "river barn wall" are different entities with identical
    // token sets, so shared adjacent bigrams rank the right origin
    // above its anagram, and a tie even there is genuine ambiguity.
    const std::set<std::string> question_bigrams = bigrams_of(ordered);

    auto order_overlap = [&question_bigrams](const std::string& key) {
        std::vector<std::string> folded;
        for (const std::string& token : raw_lower_tokens(key))
            folded.push_back(normalize_token(token));
        const std::set<std::string> key_bigrams = bigrams_of(folded);
        int shared = 0;
        for (const std::string& bigram : key_bigrams)
            if (question_bigrams.count(bigram)) ++shared;
        return shared;
    };

    struct Candidate {
        int order_score = 0;
        double strength = 0.0;
        int negative_length = 0;
        std::string node_key;
        std::string origin;
    };
    std::vector<Candidate> contenders;
    bool have_best = false;
    Candidate best;

    for (const std::string& node_key : nodes.order) {
        Node& node = *nodes.find(node_key);
        for (const std::string& origin : node.origin_order) {
            const Lineage& entry = node.origins.at(origin);
            if (entry.path.empty()) continue;
            std::map<std::string, double> combined = entry.sources;
            for (const auto& source : node.sources) {
                auto at = combined.find(source.first);
                if (at == combined.end() || at->second < source.second)
                    combined[source.first] = source.second;
            }
            std::set<std::string> covered;
            for (const auto& item : combined) covered.insert(item.first);
            if (covered != question_tokens) continue;
            // Origin coverage: the first fact of the chain may have at
            // most ONE token the question did not say - the relation
            // slot. Two absent tokens means the origin names a MORE
            // SPECIFIC subject than asked, which is how a dropped
            // qualifier silently substitutes one entity for another.
            const std::string origin_key = entry.path.front().first;
            int absent = 0;
            for (const std::string& token : fold(origin_key))
                if (!question_tokens.count(token)) ++absent;
            if (absent > 1) continue;
            const int order_score = order_overlap(origin_key);
            // The absolute form of the order rule: a multi-word subject
            // must share at least one adjacent bigram with the
            // question, because an anagram with no competitor would
            // otherwise win unopposed.
            if (raw_lower_tokens(origin_key).size() >= 3 && order_score < 1)
                continue;
            double weakest = 0.0;
            bool first = true;
            for (const auto& item : combined) {
                if (first || item.second < weakest) weakest = item.second;
                first = false;
            }
            Candidate candidate;
            candidate.order_score = order_score;
            candidate.strength = weakest;
            candidate.negative_length = -static_cast<int>(entry.path.size());
            candidate.node_key = node.key;
            candidate.origin = origin;
            contenders.push_back(candidate);
            const bool better = !have_best
                || candidate.order_score > best.order_score
                || (candidate.order_score == best.order_score
                    && candidate.strength > best.strength)
                || (candidate.order_score == best.order_score
                    && candidate.strength == best.strength
                    && candidate.negative_length > best.negative_length);
            if (better) {
                best = candidate;
                have_best = true;
            }
        }
    }
    if (!have_best) return nothing;

    // Anagram ambiguity: two origins with DIFFERENT subjects at the
    // same rank cannot be told apart, and guessing between them is the
    // pun the whole rule stack exists to refuse.
    std::set<std::string> top_subjects;
    for (const Candidate& candidate : contenders) {
        if (candidate.order_score == best.order_score
            && candidate.strength == best.strength
            && candidate.negative_length == best.negative_length) {
            Node& node = *nodes.find(candidate.node_key);
            top_subjects.insert(
                node.origins.at(candidate.origin).path.front().first);
        }
    }
    if (top_subjects.size() > 1) return nothing;

    Node& node = *nodes.find(best.node_key);
    const Lineage& entry = node.origins.at(best.origin);
    std::vector<std::pair<std::string, std::string>> premises = entry.path;
    premises.push_back({node.key, node.record.value});

    double confidence = 0.0;
    bool first_premise = true;
    for (const auto& premise : premises) {
        double value = 0.0;
        if (facts.has(premise.first)) {
            value = facts.data.at(premise.first).confidence;
        } else {
            const Fact* held = memory.recall_fact(premise.first);
            if (held != nullptr) value = held->confidence;
        }
        if (first_premise || value < confidence) confidence = value;
        first_premise = false;
    }

    const std::string subject_key = premises.front().first;
    std::vector<std::string> subject_parts;
    for (const std::string& token : raw_lower_tokens(subject_key))
        if (question_tokens.count(normalize_token(token)))
            subject_parts.push_back(token);
    const std::string subject = join_with(subject_parts, " ");
    const std::set<std::string> subject_tokens = fold(subject_key);
    std::vector<std::string> asked_parts;
    for (const std::string& token : question_tokens)
        if (!subject_tokens.count(token)) asked_parts.push_back(token);
    const std::string asked = join_with(asked_parts, " ");
    std::vector<std::string> via_parts;
    for (std::size_t i = 0; i + 1 < premises.size(); ++i)
        via_parts.push_back(premises[i].second);
    const std::string via = join_with(via_parts, ", ");

    Inference out;
    out.present = true;
    out.negated = node.record.negated;
    const std::string value = node.record.value;
    if (out.negated) {
        // The terminal is a denial: the chain earned "believed not X",
        // and the premise line reads the polarity too.
        premises.back().second = "not " + value;
    }
    const std::string spoken = out.negated ? "believed not " + value : value;
    out.answer = "the " + subject + " " + asked + " is " + spoken
        + ", via " + via;
    out.premises = premises;
    out.confidence = confidence;
    out.kind = "chain";
    out.has_conclusion = true;
    out.conclusion_key = subject + " " + asked;
    out.conclusion_value = value;
    return out;
}

}  // namespace

namespace {

bool combine_word(const std::string& token) {
    static const std::set<std::string> words = {
        "sum", "total", "together", "combined", "difference", "gap",
        "taller", "longer", "bigger", "larger", "higher", "heavier",
        "shorter", "smaller", "lower", "lighter",
        "greater", "more", "less", "fewer",
    };
    return words.count(token) != 0;
}

// A value that names a QUANTITY names no subject to ask about, so
// curiosity does not flow through it.
bool looks_numeric(const std::string& value) {
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] >= '0' && value[i] <= '9') return true;
    }
    return false;
}

bool library_unknown(const std::string& token, const Memory& memory) {
    for (const std::string& key : memory.find_facts(token, 3)) {
        const std::set<std::string> folded = fold(key);
        if (folded.count(token)) return false;
    }
    return true;
}

}  // namespace

Inference infer(const std::string& text, const Memory& memory) {
    Inference nothing;
    const std::set<std::string> question_tokens = fold(text);
    if (question_tokens.size() < 2) return nothing;
    const std::vector<std::string> ordered = ordered_fold(text);
    const Inference direct = spread(question_tokens, memory, ordered);
    if (direct.present) return direct;

    // Adjective tolerance, on the INFERENCE path only. Dropping
    // library-unknown tokens and retrying is safe HERE because
    // convergence still demands a bridge; the same drop on the recall
    // path would assert steel's number for tungsten. Up to three
    // tokens may decorate, behind the residual floor: at least two
    // known informative tokens must survive, so a question that is
    // mostly unknown refuses rather than being read away.
    std::set<std::string> unknown;
    for (const std::string& token : question_tokens)
        if (library_unknown(token, memory)) unknown.insert(token);
    std::set<std::string> residual;
    for (const std::string& token : question_tokens)
        if (!unknown.count(token)) residual.insert(token);
    if (unknown.empty() || unknown.size() > kMaxModifiers
        || residual.size() < 2)
        return nothing;

    std::vector<std::string> kept;
    for (const std::string& token : ordered)
        if (!unknown.count(token)) kept.push_back(token);
    Inference tolerated = spread(residual, memory, kept);
    if (!tolerated.present) return nothing;

    std::vector<std::string> named;
    for (const std::string& token : ordered)
        if (unknown.count(token)) named.push_back(token);
    for (const std::string& token : unknown)
        if (std::find(named.begin(), named.end(), token) == named.end())
            named.push_back(token);
    if (named.size() == 1) {
        tolerated.answer += " (reading '" + named.front()
            + "' as a modifier)";
    } else {
        std::vector<std::string> quoted;
        for (const std::string& token : named)
            quoted.push_back("'" + token + "'");
        tolerated.answer += " (reading " + join_with(quoted, " and ")
            + " as modifiers)";
    }
    return tolerated;
}

MissingPremise missing_premise(const std::string& text,
                               const Memory& memory) {
    // The metacognitive half of the spread: when convergence fails,
    // the partial activations say exactly what was missing. Curiosity
    // only flows FORWARD along a held bridge, and that constraint is
    // the spam guard - a system that asks about everything it fails to
    // answer is noise wearing a curious face.
    MissingPremise nothing;
    for (const std::string& token : raw_lower_tokens(text)) {
        if (combine_word(token)) return nothing;   // combines re-derive
    }
    std::set<std::string> question_tokens = fold(text);
    if (question_tokens.size() < 2) return nothing;

    // The one-unknown rule cannot work here, because the asked-about
    // attribute is ITSELF library-unknown - that is why it is missing.
    // The discriminator is positional: an adjective PRECEDES a
    // key-known token, while the asked attribute precedes nothing
    // known.
    std::vector<std::string> raw_sequence;
    for (const std::string& token : raw_lower_tokens(text))
        raw_sequence.push_back(normalize_token(token));
    std::set<std::string> droppable;
    for (std::size_t index = 0; index < raw_sequence.size(); ++index) {
        const std::string& folded = raw_sequence[index];
        if (!question_tokens.count(folded)) continue;
        if (!library_unknown(folded, memory)) continue;
        std::string following;
        bool found_following = false;
        for (std::size_t later = index + 1; later < raw_sequence.size();
             ++later) {
            if (question_tokens.count(raw_sequence[later])) {
                following = raw_sequence[later];
                found_following = true;
                break;
            }
        }
        if (found_following && !library_unknown(following, memory))
            droppable.insert(folded);
    }
    if (droppable.size() == 1 && question_tokens.size() >= 3) {
        for (const std::string& token : droppable)
            question_tokens.erase(token);
    }

    std::vector<std::string> probes;
    probes.push_back(join_with(
        std::vector<std::string>(question_tokens.begin(),
                                 question_tokens.end()), " "));
    for (const std::string& token : question_tokens)
        probes.push_back(token);
    const FactSet facts = reachable_facts(memory, probes);

    bool have_best = false;
    int best_covered = 0;
    int best_value_size = 0;
    std::string best_key;
    MissingPremise best;
    for (const std::string& key : facts.order) {
        const Fact& record = facts.data.at(key);
        // "not steel" once minted the premise key "not steel steel";
        // a denial names no bridge to ask through.
        if (record.negated) continue;
        const std::set<std::string> key_tokens = fold(key);
        std::set<std::string> covered, remainder;
        for (const std::string& token : key_tokens)
            if (question_tokens.count(token)) covered.insert(token);
        for (const std::string& token : question_tokens)
            if (!key_tokens.count(token)) remainder.insert(token);
        if (covered.empty() || remainder.empty()) continue;
        // A remainder wider than two tokens is a clause, not a fact
        // key: decomposition owns compounds, curiosity owns single
        // gaps.
        if (remainder.size() > 2) continue;
        const std::string value = record.value;
        Quantity ignored;
        if (looks_numeric(value)) continue;   // a quantity names no subject
        const std::set<std::string> value_tokens = fold(value);
        if (value_tokens.empty() || value_tokens.size() > 2) continue;
        int absent = 0;
        for (const std::string& token : key_tokens)
            if (!covered.count(token)) ++absent;
        if (absent > 1) continue;
        const int covered_size = static_cast<int>(covered.size());
        const int value_size = static_cast<int>(value_tokens.size());
        const bool better = !have_best
            || covered_size > best_covered
            || (covered_size == best_covered && value_size < best_value_size)
            || (covered_size == best_covered && value_size == best_value_size
                && key > best_key);
        if (!better) continue;
        std::vector<std::string> ordered_remainder;
        std::set<std::string> seen;
        for (const std::string& token : raw_lower_tokens(text)) {
            const std::string folded = normalize_token(token);
            if (remainder.count(folded) && !seen.count(folded)) {
                ordered_remainder.push_back(token);
                seen.insert(folded);
            }
        }
        std::string lowered_value;
        for (char c : value)
            lowered_value.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(c))));
        std::vector<std::string> parts{lowered_value};
        for (const std::string& token : ordered_remainder)
            parts.push_back(token);
        best.present = true;
        best.premise_key = join_with(parts, " ");
        best.via_key = key;
        best.via_value = value;
        best.original = text;
        best_covered = covered_size;
        best_value_size = value_size;
        best_key = key;
        have_best = true;
    }
    return have_best ? best : nothing;
}

}  // namespace uq
