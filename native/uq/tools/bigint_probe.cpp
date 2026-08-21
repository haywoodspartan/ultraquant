// Reads one operation per line and prints the result, so the Python
// tier can be the oracle: every answer here is checked against
// Python's own arbitrary-precision integers rather than against a
// second implementation of my own opinions.
//
//   add A B | sub A B | mul A B | fdiv A B | fmod A B
//   pow A n | iroot A n | gcd A B | str A
#include "uq/bigint.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream in(line);
        std::string op, a, b;
        in >> op >> a >> b;
        try {
            const uq::BigInt left = uq::BigInt::from_decimal(a);
            if (op == "str") { std::cout << left.str() << "\n"; continue; }
            if (op == "pow") {
                std::cout << uq::BigInt::pow(left, std::stoll(b)).str() << "\n";
                continue;
            }
            if (op == "iroot") {
                std::cout << uq::BigInt::iroot(left, std::stoi(b)).str() << "\n";
                continue;
            }
            const uq::BigInt right = uq::BigInt::from_decimal(b);
            if (op == "add") std::cout << (left + right).str() << "\n";
            else if (op == "sub") std::cout << (left - right).str() << "\n";
            else if (op == "mul") std::cout << (left * right).str() << "\n";
            else if (op == "gcd")
                std::cout << uq::BigInt::gcd(left, right).str() << "\n";
            else if (op == "fdiv") {
                uq::BigInt q, r;
                uq::BigInt::floor_divmod(left, right, q, r);
                std::cout << q.str() << "\n";
            } else if (op == "fmod") {
                uq::BigInt q, r;
                uq::BigInt::floor_divmod(left, right, q, r);
                std::cout << r.str() << "\n";
            } else std::cout << "?" << "\n";
        } catch (const std::exception& error) {
            std::cout << "error:" << error.what() << "\n";
        }
    }
    return 0;
}
