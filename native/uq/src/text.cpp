#include "uq/text.hpp"

#include <cctype>

namespace uq {

std::string normalize_token(const std::string& token) {
    const std::size_t size = token.size();
    if (size < 4) return token;
    auto ends_with = [&token](const std::string& tail) {
        return token.size() >= tail.size()
            && token.compare(token.size() - tail.size(), tail.size(),
                             tail) == 0;
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

bool informative(const std::string& token) {
    // Generated from the Python tier's own list and pinned against
    // it, so "copied rather than reinvented" is a claim a test can
    // check rather than a comment nobody re-reads.
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

std::set<std::string> folded_tokens(const std::string& text) {
    std::set<std::string> out;
    for (const std::string& token : raw_tokens(text))
        if (informative(token)) out.insert(normalize_token(token));
    return out;
}

std::vector<std::string> ordered_tokens(const std::string& text) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const std::string& token : raw_tokens(text)) {
        const std::string folded = normalize_token(token);
        if (seen.count(folded) || !informative(token)) continue;
        seen.insert(folded);
        out.push_back(folded);
    }
    return out;
}

}  // namespace uq
