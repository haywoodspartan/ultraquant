// Arbitrary-precision integers, from scratch - the native tier's
// answer to a Python int.
//
// The pure-Python tier defines this system's semantics and every
// other tier reproduces them; that rule is what makes a native
// variant worth having and also what makes it hard here. Python's
// int is arbitrary precision, so "what is 2 ^ 1000?" is a 302-digit
// answer, an exact rational's denominator grows without asking, and
// a root enclosure scales its radicand by 10^(places*degree) before
// taking an integer root. A 64-bit port would diverge from the
// semantics on the first interesting question and be quietly wrong
// on the rest, which is worse than being slower.
//
// So: sign-magnitude, base 1,000,000,000, limbs least-significant
// first. Base 1e9 rather than 2^32 because every decimal boundary in
// this system - parsing "0.1" exactly, printing an exact decimal,
// scaling by a power of ten for rounding - becomes limb arithmetic
// instead of repeated division, and those paths are the ones that
// have to be exactly right rather than merely fast.
#ifndef UQ_BIGINT_HPP
#define UQ_BIGINT_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace uq {

class BigInt {
public:
    BigInt() = default;
    BigInt(long long value);
    explicit BigInt(const std::string& decimal);

    static BigInt from_decimal(const std::string& decimal);

    bool is_zero() const { return limbs_.empty(); }
    bool negative() const { return negative_; }
    int sign() const { return limbs_.empty() ? 0 : (negative_ ? -1 : 1); }

    BigInt operator-() const;
    BigInt operator+(const BigInt& other) const;
    BigInt operator-(const BigInt& other) const;
    BigInt operator*(const BigInt& other) const;
    BigInt operator/(const BigInt& other) const;   // truncating
    BigInt operator%(const BigInt& other) const;   // sign of dividend

    BigInt& operator+=(const BigInt& other) { return *this = *this + other; }
    BigInt& operator-=(const BigInt& other) { return *this = *this - other; }
    BigInt& operator*=(const BigInt& other) { return *this = *this * other; }

    bool operator==(const BigInt& other) const;
    bool operator!=(const BigInt& other) const { return !(*this == other); }
    bool operator<(const BigInt& other) const;
    bool operator<=(const BigInt& other) const { return !(other < *this); }
    bool operator>(const BigInt& other) const { return other < *this; }
    bool operator>=(const BigInt& other) const { return !(*this < other); }

    // Floor division and its remainder - the pair Python's // and %
    // give, which is NOT what C++ integer division gives for
    // negatives. Every rounding decision in this system is specified
    // in terms of Python's floor, so the native tier has to have it
    // under its own name rather than hope the difference never shows.
    static void floor_divmod(const BigInt& top, const BigInt& bottom,
                             BigInt& quotient, BigInt& remainder);
    static BigInt floor_div(const BigInt& top, const BigInt& bottom);

    static BigInt gcd(BigInt left, BigInt right);
    static BigInt pow(const BigInt& base, long long exponent);
    static BigInt pow10(long long exponent);

    // The integer part of the degree-th root, exactly, by binary
    // search on integers - no float ever touches a value here, which
    // is the whole reason roots are trustworthy in this system.
    static BigInt iroot(const BigInt& value, int degree);

    BigInt abs() const;
    std::string str() const;
    long long to_ll() const;            // undefined if it does not fit
    bool fits_ll() const;

private:
    static const std::uint32_t kBase = 1000000000u;
    static const int kDigits = 9;

    std::vector<std::uint32_t> limbs_;   // little-endian, base 1e9
    bool negative_ = false;

    void trim();
    static int compare_magnitude(const BigInt& a, const BigInt& b);
    static std::vector<std::uint32_t> add_magnitude(
        const std::vector<std::uint32_t>& a,
        const std::vector<std::uint32_t>& b);
    static std::vector<std::uint32_t> sub_magnitude(
        const std::vector<std::uint32_t>& a,
        const std::vector<std::uint32_t>& b);
    static void divmod_magnitude(const BigInt& a, const BigInt& b,
                                 BigInt& q, BigInt& r);
};

}  // namespace uq

#endif  // UQ_BIGINT_HPP
