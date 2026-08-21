// The fact store, natively - the semantic half of memory/systematic.py.
//
// Facts are the substrate every question form in this system stands
// on, so the parity obligation is heavier here than it looks: not
// just "the same value comes back", but the same OUTCOME of storing
// (new, reinforced, revised), the same spoken form of a replaced
// belief, the same confidence after a reinforcement, and the same
// ranking when retrieval has to choose. Polarity is part of a fact's
// identity - "not steel" after "steel" is a change of mind, never a
// reinforcement - and that rule lives in the store rather than above
// it, so it cannot be forgotten by a caller.
//
// The shard layer is deliberately NOT ported. The Python tier pages
// buckets from disk because a 1.2T-parameter library cannot sit in
// RAM; a native tier that reproduced the paging would be reproducing
// a storage decision, not a semantic one, and §11.86 already
// measured what that layer costs and why it is the cache rather than
// the lookup that matters. What is reproduced is what a caller can
// observe about beliefs.
#ifndef UQ_MEMORY_HPP
#define UQ_MEMORY_HPP

#include <map>
#include <string>
#include <vector>

namespace uq {

struct Fact {
    std::string value;
    double confidence = 0.0;
    long long reinforcements = 0;
    bool negated = false;
};

struct Episode {
    std::string kind;                 // "revision" or "retraction"
    std::string key;
    std::string old_value;            // revisions
    std::string new_value;            // revisions
    std::string because;              // retractions
    std::vector<std::string> tags;
};

// What storing a fact DID - the surface says what changed instead of
// noting a change silently, so this is part of the contract.
struct StoreOutcome {
    std::string outcome;              // "new" | "reinforced" | "revised"
    std::string was;                  // revisions: the old belief, spoken
    std::vector<std::string> retracted;
};

class Memory {
public:
    StoreOutcome remember_fact(const std::string& key,
                               const std::string& value,
                               double confidence = 0.5,
                               bool negated = false);

    bool confirm_fact(const std::string& key, double confidence = 0.9);

    const Fact* recall_fact(const std::string& key) const;
    std::vector<std::string> fact_keys() const;
    std::size_t size() const { return facts_.size(); }

    // Keyword overlap over stored keys, with the §11.44 tie-break:
    // equal overlap resolves toward the re-attested fact, never past
    // a better token match.
    std::vector<std::string> find_facts(const std::string& text,
                                        std::size_t top_k = 5) const;

    const std::vector<Episode>& episodes() const { return episodes_; }

    // A derived fact records what it rested on, so a revised premise
    // can take its conclusions down recursively.
    void consolidate_fact(const std::string& key, const std::string& value,
                          double confidence,
                          const std::vector<std::string>& premises,
                          bool negated = false);

private:
    std::map<std::string, Fact> facts_;
    std::map<std::string, std::vector<std::string>> derived_from_;
    std::vector<Episode> episodes_;

    std::vector<std::string> retract_derivatives(const std::string& key);
};

std::vector<std::string> tokens_of(const std::string& text);

}  // namespace uq

#endif  // UQ_MEMORY_HPP
