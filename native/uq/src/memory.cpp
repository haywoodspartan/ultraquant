#include "uq/memory.hpp"

#include <algorithm>
#include <cctype>

#include "uq/calculate.hpp"   // normalize_token

namespace uq {

namespace {

// Raw [a-z0-9]+ over lowercased text - NO plural fold. That is not
// an oversight to be tidied up later: the store's own retrieval
// compares raw tokens, and the fold belongs to the router and the
// interpreter above it. A native tier that folded here would find
// facts the Python store does not, which is the same kind of wrong
// as missing them.
std::vector<std::string> raw_tokens(const std::string& text) {
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

}  // namespace

std::vector<std::string> tokens_of(const std::string& text) {
    // The folded form, for the callers above the store that do fold.
    std::vector<std::string> out;
    for (const std::string& token : raw_tokens(text))
        out.push_back(normalize_token(token));
    return out;
}

StoreOutcome Memory::remember_fact(const std::string& key,
                                   const std::string& value,
                                   double confidence, bool negated) {
    StoreOutcome result;
    auto found = facts_.find(key);
    if (found == facts_.end()) {
        Fact fact;
        fact.value = value;
        fact.confidence = confidence;
        fact.reinforcements = 0;
        fact.negated = negated;
        facts_[key] = fact;
        result.outcome = "new";
        return result;
    }
    Fact& fact = found->second;
    if (fact.value == value && fact.negated == negated) {
        fact.confidence = std::min(1.0, fact.confidence + 0.1);
        fact.reinforcements += 1;
        result.outcome = "reinforced";
        return result;
    }
    // Polarity is part of a fact's identity, so a flip lands here and
    // not in the branch above: "not steel" after "steel" is a change
    // of mind, never a reinforcement.
    const std::string was = fact.negated ? "not " + fact.value : fact.value;
    fact.value = value;
    fact.negated = negated;
    fact.confidence = confidence;
    derived_from_.erase(key);
    const std::vector<std::string> retracted = retract_derivatives(key);
    for (const std::string& gone : retracted) {
        Episode episode;
        episode.kind = "retraction";
        episode.key = gone;
        episode.because = "premise '" + key + "' was revised";
        episode.tags = {"fact", gone};
        episodes_.push_back(episode);
    }
    Episode episode;
    episode.kind = "revision";
    episode.key = key;
    episode.old_value = was;
    episode.new_value = negated ? "not " + value : value;
    episode.tags = {"fact", key};
    episodes_.push_back(episode);
    result.outcome = "revised";
    result.was = was;
    result.retracted = retracted;
    return result;
}

std::vector<std::string> Memory::retract_derivatives(const std::string& key) {
    // A stack, not a sweep, and the ORDER is part of the contract:
    // the Python tier walks a stack of revised keys and appends each
    // casualty as it finds it, scanning keys in sorted order. The
    // retracted list is spoken aloud when a belief changes, so two
    // tiers that drop the same facts in a different order have not
    // said the same thing.
    std::vector<std::string> gone;
    std::vector<std::string> stack{key};
    while (!stack.empty()) {
        const std::string changed = stack.back();
        stack.pop_back();
        for (const std::string& candidate : fact_keys()) {
            const auto found = derived_from_.find(candidate);
            if (found == derived_from_.end()) continue;
            const bool rests_on =
                std::find(found->second.begin(), found->second.end(), changed)
                != found->second.end();
            if (!rests_on) continue;
            facts_.erase(candidate);
            derived_from_.erase(candidate);
            gone.push_back(candidate);
            stack.push_back(candidate);
        }
    }
    return gone;
}

void Memory::consolidate_fact(const std::string& key,
                              const std::string& value, double confidence,
                              const std::vector<std::string>& premises,
                              bool negated) {
    // The record is written WHOLE rather than routed through
    // remember_fact: a consolidation is not a statement, so it logs
    // no revision episode and resets the reinforcement count. The
    // provenance is load-bearing - a derived fact whose premise is
    // later revised must come down with it.
    Fact fact;
    fact.value = value;
    fact.confidence = confidence;
    fact.reinforcements = 0;
    fact.negated = negated;
    facts_[key] = fact;
    derived_from_[key] = premises;
}

bool Memory::confirm_fact(const std::string& key, double confidence) {
    auto found = facts_.find(key);
    if (found == facts_.end()) return false;
    // Confidence is SET, not raised: being told "yes, that is
    // correct" is direct testimony rather than another passing
    // mention, and it can lower a confidence as well as raise one.
    // The reinforcement count still moves, because the fact was
    // attested again - and that count is what §11.44's retrieval
    // tie-break reads, so getting this wrong changes which fact
    // answers a question.
    found->second.confidence = std::max(0.0, std::min(1.0, confidence));
    found->second.reinforcements += 1;
    return true;
}

const Fact* Memory::recall_fact(const std::string& key) const {
    const auto found = facts_.find(key);
    return found == facts_.end() ? nullptr : &found->second;
}

std::vector<std::string> Memory::fact_keys() const {
    std::vector<std::string> keys;
    keys.reserve(facts_.size());
    for (const auto& entry : facts_) keys.push_back(entry.first);
    return keys;   // std::map is sorted, which is what sorted() gives
}

std::vector<std::string> Memory::find_facts(const std::string& text,
                                            std::size_t top_k) const {
    const std::vector<std::string> wanted_list = raw_tokens(text);
    std::vector<std::string> wanted;
    for (const std::string& token : wanted_list) {
        if (std::find(wanted.begin(), wanted.end(), token) == wanted.end())
            wanted.push_back(token);
    }
    struct Scored { long long overlap; double weight; std::string key; };
    std::vector<Scored> scored;
    for (const auto& entry : facts_) {
        const std::vector<std::string> key_tokens = raw_tokens(entry.first);
        long long overlap = 0;
        for (const std::string& token : wanted) {
            if (std::find(key_tokens.begin(), key_tokens.end(), token)
                != key_tokens.end())
                ++overlap;
        }
        if (overlap == 0) continue;
        // §11.44's tie-break: equal overlap resolves toward the fact
        // that has been re-attested, and never past a better match.
        scored.push_back({overlap,
                          static_cast<double>(entry.second.reinforcements)
                              + entry.second.confidence,
                          entry.first});
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [](const Scored& a, const Scored& b) {
                         if (a.overlap != b.overlap) return a.overlap > b.overlap;
                         if (a.weight != b.weight) return a.weight > b.weight;
                         return a.key < b.key;
                     });
    std::vector<std::string> out;
    for (const Scored& item : scored) {
        if (out.size() >= top_k) break;
        out.push_back(item.key);
    }
    return out;
}

}  // namespace uq
