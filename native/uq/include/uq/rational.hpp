// Exact rationals - the native tier's Fraction.
//
// §11.76 chose exact rational arithmetic over floats for one reason:
// a system whose discipline is "say exactly what is true" cannot
// answer a question about tenths with 0.30000000000000004. The
// native tier inherits that choice rather than re-deciding it, which
// means it inherits the obligations too - decimals parse exactly
// ("0.1" is one tenth, not the nearest double), values stay
// normalised so equality is structural, and nothing is ever rounded
// unless somebody asked.
#ifndef UQ_RATIONAL_HPP
#define UQ_RATIONAL_HPP

#include <string>

#include "uq/bigint.hpp"

namespace uq {

class Rational {
public:
    Rational() : top_(0), bottom_(1) {}
    Rational(long long value) : top_(value), bottom_(1) {}
    Rational(BigInt top, BigInt bottom);

    // "3", "-1.75", "0.1", "7/3", "1e-06" - every spelling the
    // Python tier can hand across, read exactly.
    static Rational parse(const std::string& text);

    const BigInt& top() const { return top_; }
    const BigInt& bottom() const { return bottom_; }

    bool is_zero() const { return top_.is_zero(); }
    bool negative() const { return top_.negative(); }
    bool is_integer() const { return bottom_ == BigInt(1); }

    Rational operator-() const;
    Rational operator+(const Rational& other) const;
    Rational operator-(const Rational& other) const;
    Rational operator*(const Rational& other) const;
    Rational operator/(const Rational& other) const;   // throws on zero

    bool operator==(const Rational& other) const;
    bool operator!=(const Rational& other) const { return !(*this == other); }
    bool operator<(const Rational& other) const;
    bool operator<=(const Rational& other) const { return !(other < *this); }
    bool operator>(const Rational& other) const { return other < *this; }
    bool operator>=(const Rational& other) const { return !(*this < other); }

    Rational pow(long long exponent) const;
    BigInt floor() const;

private:
    BigInt top_;
    BigInt bottom_;   // always positive, always coprime with top_

    void normalise();
};

}  // namespace uq

#endif  // UQ_RATIONAL_HPP
