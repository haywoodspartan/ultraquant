// One question per line in, one canonical record per line out, so
// the Python tier can be the oracle for the whole reader rather than
// for a handful of hand-picked cases.
//
//   none                                     - not an arithmetic question
//   refuse|<refusal>                         - a question with no answer
//   bounds|<expression>|<low>|<high>         - an enclosed root
//   ok|<expression>|<shown>|<frac>|<places>|<rounded>|<exact>
#include "uq/calculate.hpp"

#include <iostream>
#include <string>

namespace {

std::string record(const uq::MathResult& result) {
    if (!result.present) return "none";
    if (!result.refusal.empty()) return "refuse|" + result.refusal;
    if (result.has_bounds)
        return "bounds|" + result.expression + "|" + result.low + "|"
             + result.high;
    return "ok|" + result.expression + "|" + result.shown + "|"
         + (result.fractional ? "1" : "0") + "|"
         + (result.rounded_to < 0 ? "-" : std::to_string(result.rounded_to))
         + "|" + (result.was_rounded ? "1" : "0") + "|" + result.exact_shown;
}

}  // namespace

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // A list question and an expression question are different
        // readers in the Python tier too; the caller asks for one or
        // the other by prefix so the parity check stays honest about
        // which door it went through.
        try {
            if (line.rfind("list ", 0) == 0)
                std::cout << record(uq::read_list(line.substr(5))) << "\n";
            else
                std::cout << record(uq::evaluate(line)) << "\n";
        } catch (const std::exception& error) {
            std::cout << "crash|" << error.what() << "\n";
        }
    }
    return 0;
}
