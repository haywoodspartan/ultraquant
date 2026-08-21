// A scripted store, then questions put to the inference engine
// alone, so a chain can be compared without the pipeline's other
// branches in the way.
//
//   remember <conf> <neg> <key>|<value>
//   infer <question>       -> infer|<describe()>  or  infer|none
//   gap <question>         -> gap|<premise_key>|<via_key>|<via_value>
#include "uq/inference.hpp"

#include <iostream>
#include <sstream>

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
            memory.remember_fact(rest.substr(0, bar), rest.substr(bar + 1),
                                 confidence, negated != 0);
            std::cout << "ok" << std::endl;
        } else if (op == "infer") {
            std::string question;
            std::getline(in, question);
            if (!question.empty() && question.front() == ' ')
                question.erase(0, 1);
            const uq::Inference result = uq::infer(question, memory);
            if (!result.present) std::cout << "infer|none" << std::endl;
            else std::cout << "infer|" << result.describe() << std::endl;
        } else if (op == "gap") {
            std::string question;
            std::getline(in, question);
            if (!question.empty() && question.front() == ' ')
                question.erase(0, 1);
            const uq::MissingPremise gap =
                uq::missing_premise(question, memory);
            if (!gap.present) std::cout << "gap|none" << std::endl;
            else std::cout << "gap|" << gap.premise_key << "|"
                           << gap.via_key << "|" << gap.via_value
                           << std::endl;
        } else {
            std::cout << "?" << std::endl;
        }
    }
    return 0;
}
