// A scripted fact store: one command per line in, one record out, so
// the Python store can be the oracle for storing as well as for
// recalling.
//
//   remember <conf> <neg> <key>|<value>   -> outcome record
//   confirm <conf> <key>                  -> ok|1/0
//   recall <key>                          -> fact|<value>|<conf>|<neg>|<reinf>
//   find <top_k> <text>                   -> found|k1;k2;...
//   keys                                  -> keys|k1;k2;...
//   episodes                              -> ep|kind:key:old:new;...
#include "uq/memory.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::string join(const std::vector<std::string>& items, char sep) {
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out.push_back(sep);
        out += items[i];
    }
    return out;
}

std::string number(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

}  // namespace

int main() {
    uq::Memory memory;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream in(line);
        std::string op;
        in >> op;
        if (op == "remember") {
            double confidence = 0.5;
            int negated = 0;
            in >> confidence >> negated;
            std::string rest;
            std::getline(in, rest);
            if (!rest.empty() && rest.front() == ' ') rest.erase(0, 1);
            const std::size_t bar = rest.find('|');
            const std::string key = rest.substr(0, bar);
            const std::string value = rest.substr(bar + 1);
            const uq::StoreOutcome result =
                memory.remember_fact(key, value, confidence, negated != 0);
            std::cout << result.outcome << "|" << result.was << "|"
                      << join(result.retracted, ';') << "\n";
        } else if (op == "confirm") {
            double confidence = 0.9;
            in >> confidence;
            std::string key;
            std::getline(in, key);
            if (!key.empty() && key.front() == ' ') key.erase(0, 1);
            std::cout << "ok|" << (memory.confirm_fact(key, confidence) ? 1 : 0)
                      << "\n";
        } else if (op == "recall") {
            std::string key;
            std::getline(in, key);
            if (!key.empty() && key.front() == ' ') key.erase(0, 1);
            const uq::Fact* fact = memory.recall_fact(key);
            if (fact == nullptr) std::cout << "absent\n";
            else std::cout << "fact|" << fact->value << "|"
                           << number(fact->confidence) << "|"
                           << (fact->negated ? 1 : 0) << "|"
                           << fact->reinforcements << "\n";
        } else if (op == "find") {
            std::size_t top_k = 5;
            in >> top_k;
            std::string text;
            std::getline(in, text);
            if (!text.empty() && text.front() == ' ') text.erase(0, 1);
            std::cout << "found|" << join(memory.find_facts(text, top_k), ';')
                      << "\n";
        } else if (op == "keys") {
            std::cout << "keys|" << join(memory.fact_keys(), ';') << "\n";
        } else if (op == "consolidate") {
            double confidence = 0.5;
            in >> confidence;
            std::string rest;
            std::getline(in, rest);
            if (!rest.empty() && rest.front() == ' ') rest.erase(0, 1);
            // key|value|premise;premise
            const std::size_t first = rest.find('|');
            const std::size_t second = rest.find('|', first + 1);
            const std::string key = rest.substr(0, first);
            const std::string value =
                rest.substr(first + 1, second - first - 1);
            std::vector<std::string> premises;
            std::string tail = rest.substr(second + 1);
            std::size_t at = 0;
            while (at <= tail.size() && !tail.empty()) {
                const std::size_t next = tail.find(';', at);
                premises.push_back(tail.substr(at, next - at));
                if (next == std::string::npos) break;
                at = next + 1;
            }
            memory.consolidate_fact(key, value, confidence, premises);
            std::cout << "consolidated\n";
        } else if (op == "episodes") {
            // Most recent FIRST, which is what recall_episodes gives:
            // a history answer reads the newest change first, so the
            // order is semantics rather than storage.
            std::vector<std::string> spoken;
            const std::vector<uq::Episode>& all = memory.episodes();
            for (auto it = all.rbegin(); it != all.rend(); ++it) {
                const uq::Episode& episode = *it;
                spoken.push_back(episode.kind + ":" + episode.key + ":"
                                 + episode.old_value + ":"
                                 + episode.new_value + ":"
                                 + episode.because);
            }
            std::cout << "ep|" << join(spoken, ';') << "\n";
        } else {
            std::cout << "?\n";
        }
    }
    return 0;
}
