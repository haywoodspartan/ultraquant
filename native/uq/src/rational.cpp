#include "uq/rational.hpp"

#include <stdexcept>

namespace uq {

void Rational::normalise() {
    if (bottom_.is_zero()) throw std::domain_error("zero denominator");
    if (bottom_.negative()) {
        top_ = -top_;
        bottom_ = -bottom_;
    }
    const BigInt divisor = BigInt::gcd(top_, bottom_);
    if (!divisor.is_zero() && divisor != BigInt(1)) {
        top_ = top_ / divisor;
        bottom_ = bottom_ / divisor;
    }
    if (top_.is_zero()) bottom_ = BigInt(1);
}

Rational::Rational(BigInt top, BigInt bottom)
    : top_(std::move(top)), bottom_(std::move(bottom)) {
    normalise();
}

Rational Rational::parse(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("empty number");
    const std::size_t slash = text.find('/');
    if (slash != std::string::npos) {
        return Rational(BigInt::from_decimal(text.substr(0, slash)),
                        BigInt::from_decimal(text.substr(slash + 1)));
    }
    // A decimal, and possibly an exponent: "1e-06" is what
    // str(1e-06) gives Python, and Fraction reads it exactly, so the
    // native tier has to read it the same way or the unit table
    // itself would disagree across tiers.
    std::string body = text;
    long long exponent = 0;
    const std::size_t e = body.find_first_of("eE");
    if (e != std::string::npos) {
        exponent = std::stoll(body.substr(e + 1));
        body = body.substr(0, e);
    }
    const std::size_t dot = body.find('.');
    if (dot != std::string::npos) {
        const std::string fraction = body.substr(dot + 1);
        exponent -= static_cast<long long>(fraction.size());
        body = body.substr(0, dot) + fraction;
    }
    BigInt whole = BigInt::from_decimal(body);
    if (exponent >= 0) return Rational(whole * BigInt::pow10(exponent),
                                       BigInt(1));
    return Rational(whole, BigInt::pow10(-exponent));
}

Rational Rational::operator-() const {
    Rational out;
    out.top_ = -top_;
    out.bottom_ = bottom_;
    return out;
}

Rational Rational::operator+(const Rational& other) const {
    return Rational(top_ * other.bottom_ + other.top_ * bottom_,
                    bottom_ * other.bottom_);
}

Rational Rational::operator-(const Rational& other) const {
    return *this + (-other);
}

Rational Rational::operator*(const Rational& other) const {
    return Rational(top_ * other.top_, bottom_ * other.bottom_);
}

Rational Rational::operator/(const Rational& other) const {
    if (other.is_zero()) throw std::domain_error("division by zero");
    return Rational(top_ * other.bottom_, bottom_ * other.top_);
}

bool Rational::operator==(const Rational& other) const {
    return top_ == other.top_ && bottom_ == other.bottom_;
}

bool Rational::operator<(const Rational& other) const {
    return top_ * other.bottom_ < other.top_ * bottom_;
}

Rational Rational::pow(long long exponent) const {
    if (exponent >= 0)
        return Rational(BigInt::pow(top_, exponent),
                        BigInt::pow(bottom_, exponent));
    if (top_.is_zero()) throw std::domain_error("zero to a negative power");
    return Rational(BigInt::pow(bottom_, -exponent),
                    BigInt::pow(top_, -exponent));
}

BigInt Rational::floor() const {
    return BigInt::floor_div(top_, bottom_);
}

}  // namespace uq
