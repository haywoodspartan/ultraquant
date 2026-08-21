// One line of conversation in, one intent-and-response record out.
// The native tier's CLI and the parity harness are the same program:
// what the gate compares is exactly what a person would read.
#include "uq/interpreter.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    const bool plain = argc > 1 && std::string(argv[1]) == "--plain";
    uq::Session session;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "::dump") {
            // The store, at the end, so the gate can check that a
            // pipeline which SAID the right thing also remembered the
            // right thing. Separators are printable and absent from
            // every key and value this tier stores; control
            // characters would be tidier and much harder to read in
            // a failure message.
            std::string out = "store|";
            bool first = true;
            for (const std::string& key : session.memory().fact_keys()) {
                const uq::Fact* fact = session.memory().recall_fact(key);
                if (fact == nullptr) continue;
                std::ostringstream confidence;
                confidence << std::fixed << std::setprecision(2)
                           << fact->confidence;
                if (!first) out += " ;; ";
                first = false;
                out += key + " :: " + fact->value + " :: "
                     + (fact->negated ? "1" : "0") + " :: "
                     + confidence.str();
            }
            std::cout << out << std::endl;
            continue;
        }
        try {
            const uq::Turn turn = session.run(line);
            if (plain) std::cout << turn.response << "\n";
            else std::cout << turn.intent << "|" << turn.response << "\n";
        } catch (const std::exception& error) {
            std::cout << "crash|" << error.what() << "\n";
        }
    }
    return 0;
}
