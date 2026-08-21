#include "uq/bigint.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace uq {

void BigInt::trim() {
    while (!limbs_.empty() && limbs_.back() == 0) limbs_.pop_back();
    if (limbs_.empty()) negative_ = false;   // there is one zero, and it is +0
}

BigInt::BigInt(long long value) {
    negative_ = value < 0;
    unsigned long long magnitude =
        negative_ ? (0ull - static_cast<unsigned long long>(value))
                  : static_cast<unsigned long long>(value);
    while (magnitude) {
        limbs_.push_back(static_cast<std::uint32_t>(magnitude % kBase));
        magnitude /= kBase;
    }
}

BigInt::BigInt(const std::string& decimal) { *this = from_decimal(decimal); }

BigInt BigInt::from_decimal(const std::string& decimal) {
    BigInt out;
    std::size_t at = 0;
    bool negative = false;
    if (at < decimal.size() && (decimal[at] == '+' || decimal[at] == '-')) {
        negative = decimal[at] == '-';
        ++at;
    }
    const std::string digits = decimal.substr(at);
    if (digits.empty()) throw std::invalid_argument("empty integer");
    for (char c : digits) {
        if (c < '0' || c > '9') throw std::invalid_argument("not an integer");
    }
    // Chew from the right in nine-digit bites, which is exactly one
    // limb per bite because the base was chosen to make it so.
    for (long long end = static_cast<long long>(digits.size()); end > 0;
         end -= kDigits) {
        const long long start = std::max<long long>(0, end - kDigits);
        out.limbs_.push_back(static_cast<std::uint32_t>(
            std::strtoul(digits.substr(static_cast<std::size_t>(start),
                                       static_cast<std::size_t>(end - start))
                             .c_str(), nullptr, 10)));
    }
    out.negative_ = negative;
    out.trim();
    return out;
}

int BigInt::compare_magnitude(const BigInt& a, const BigInt& b) {
    if (a.limbs_.size() != b.limbs_.size())
        return a.limbs_.size() < b.limbs_.size() ? -1 : 1;
    for (std::size_t i = a.limbs_.size(); i-- > 0;) {
        if (a.limbs_[i] != b.limbs_[i])
            return a.limbs_[i] < b.limbs_[i] ? -1 : 1;
    }
    return 0;
}

std::vector<std::uint32_t> BigInt::add_magnitude(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b) {
    std::vector<std::uint32_t> out;
    out.reserve(std::max(a.size(), b.size()) + 1);
    std::uint32_t carry = 0;
    for (std::size_t i = 0; i < a.size() || i < b.size() || carry; ++i) {
        std::uint64_t sum = carry;
        if (i < a.size()) sum += a[i];
        if (i < b.size()) sum += b[i];
        out.push_back(static_cast<std::uint32_t>(sum % kBase));
        carry = static_cast<std::uint32_t>(sum / kBase);
    }
    return out;
}

std::vector<std::uint32_t> BigInt::sub_magnitude(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b) {
    std::vector<std::uint32_t> out;
    out.reserve(a.size());
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::int64_t digit = static_cast<std::int64_t>(a[i]) - borrow
            - (i < b.size() ? static_cast<std::int64_t>(b[i]) : 0);
        if (digit < 0) { digit += kBase; borrow = 1; } else { borrow = 0; }
        out.push_back(static_cast<std::uint32_t>(digit));
    }
    return out;
}

BigInt BigInt::operator-() const {
    BigInt out = *this;
    if (!out.limbs_.empty()) out.negative_ = !out.negative_;
    return out;
}

BigInt BigInt::abs() const {
    BigInt out = *this;
    out.negative_ = false;
    return out;
}

BigInt BigInt::operator+(const BigInt& other) const {
    BigInt out;
    if (negative_ == other.negative_) {
        out.limbs_ = add_magnitude(limbs_, other.limbs_);
        out.negative_ = negative_;
    } else {
        const int order = compare_magnitude(*this, other);
        if (order == 0) return BigInt();
        if (order > 0) {
            out.limbs_ = sub_magnitude(limbs_, other.limbs_);
            out.negative_ = negative_;
        } else {
            out.limbs_ = sub_magnitude(other.limbs_, limbs_);
            out.negative_ = other.negative_;
        }
    }
    out.trim();
    return out;
}

BigInt BigInt::operator-(const BigInt& other) const { return *this + (-other); }

BigInt BigInt::operator*(const BigInt& other) const {
    if (limbs_.empty() || other.limbs_.empty()) return BigInt();
    std::vector<std::uint64_t> wide(limbs_.size() + other.limbs_.size(), 0);
    for (std::size_t i = 0; i < limbs_.size(); ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < other.limbs_.size() || carry; ++j) {
            std::uint64_t current = wide[i + j] + carry;
            if (j < other.limbs_.size())
                current += static_cast<std::uint64_t>(limbs_[i])
                    * other.limbs_[j];
            wide[i + j] = current % kBase;
            carry = current / kBase;
        }
    }
    BigInt out;
    out.limbs_.reserve(wide.size());
    for (std::uint64_t limb : wide)
        out.limbs_.push_back(static_cast<std::uint32_t>(limb));
    out.negative_ = negative_ != other.negative_;
    out.trim();
    return out;
}

void BigInt::divmod_magnitude(const BigInt& a, const BigInt& b,
                              BigInt& q, BigInt& r) {
    // Schoolbook long division over base-1e9 limbs, with the trial
    // digit found by binary search rather than estimated: slower per
    // limb and impossible to get subtly wrong, which is the right
    // trade for a tier whose only job is to agree with another one.
    q = BigInt();
    r = BigInt();
    if (b.limbs_.empty()) throw std::domain_error("division by zero");
    q.limbs_.assign(a.limbs_.size(), 0);
    const BigInt divisor = b.abs();
    for (std::size_t i = a.limbs_.size(); i-- > 0;) {
        r.limbs_.insert(r.limbs_.begin(), a.limbs_[i]);
        r.trim();
        std::uint32_t low = 0, high = kBase - 1, digit = 0;
        while (low <= high) {
            const std::uint32_t middle = low + (high - low) / 2;
            const BigInt candidate =
                divisor * BigInt(static_cast<long long>(middle));
            if (compare_magnitude(candidate, r) <= 0) {
                digit = middle;
                low = middle + 1;
            } else {
                if (middle == 0) break;
                high = middle - 1;
            }
        }
        q.limbs_[i] = digit;
        r = r - divisor * BigInt(static_cast<long long>(digit));
    }
    q.trim();
    r.trim();
}

BigInt BigInt::operator/(const BigInt& other) const {
    BigInt q, r;
    divmod_magnitude(this->abs(), other.abs(), q, r);
    q.negative_ = !q.limbs_.empty() && (negative_ != other.negative_);
    q.trim();
    return q;
}

BigInt BigInt::operator%(const BigInt& other) const {
    BigInt q, r;
    divmod_magnitude(this->abs(), other.abs(), q, r);
    r.negative_ = !r.limbs_.empty() && negative_;
    r.trim();
    return r;
}

void BigInt::floor_divmod(const BigInt& top, const BigInt& bottom,
                          BigInt& quotient, BigInt& remainder) {
    // Python floor semantics, spelled out: the quotient rounds toward
    // negative infinity and the remainder takes the DIVISOR's sign.
    // C++ truncates toward zero, and that difference is exactly the
    // class of bug that would make a rounding rule disagree across
    // tiers on negative values only - which is to say, invisibly
    // until it matters.
    BigInt q = top / bottom;
    BigInt r = top - q * bottom;
    if (!r.is_zero() && (r.negative() != bottom.negative())) {
        q = q - BigInt(1);
        r = r + bottom;
    }
    quotient = q;
    remainder = r;
}

BigInt BigInt::floor_div(const BigInt& top, const BigInt& bottom) {
    BigInt q, r;
    floor_divmod(top, bottom, q, r);
    return q;
}

bool BigInt::operator==(const BigInt& other) const {
    return negative_ == other.negative_ && limbs_ == other.limbs_;
}

bool BigInt::operator<(const BigInt& other) const {
    if (negative_ != other.negative_) return negative_;
    const int order = compare_magnitude(*this, other);
    return negative_ ? order > 0 : order < 0;
}

BigInt BigInt::gcd(BigInt left, BigInt right) {
    left = left.abs();
    right = right.abs();
    while (!right.is_zero()) {
        BigInt next = left % right;
        left = right;
        right = next;
    }
    return left;
}

BigInt BigInt::pow(const BigInt& base, long long exponent) {
    if (exponent < 0) throw std::domain_error("negative exponent");
    BigInt result(1), factor = base;
    while (exponent > 0) {
        if (exponent & 1) result = result * factor;
        exponent >>= 1;
        if (exponent) factor = factor * factor;
    }
    return result;
}

BigInt BigInt::pow10(long long exponent) { return pow(BigInt(10), exponent); }

BigInt BigInt::iroot(const BigInt& value, int degree) {
    if (value.negative()) throw std::domain_error("negative radicand");
    if (value < BigInt(2)) return value;
    BigInt low(0), high(1);
    while (pow(high, degree) <= value) high = high * BigInt(2);
    while (low < high) {
        const BigInt middle = floor_div(low + high + BigInt(1), BigInt(2));
        if (pow(middle, degree) <= value) low = middle;
        else high = middle - BigInt(1);
    }
    return low;
}

std::string BigInt::str() const {
    if (limbs_.empty()) return "0";
    std::string out = negative_ ? "-" : "";
    out += std::to_string(limbs_.back());
    for (std::size_t i = limbs_.size() - 1; i-- > 0;) {
        const std::string piece = std::to_string(limbs_[i]);
        out += std::string(static_cast<std::size_t>(kDigits) - piece.size(), '0');
        out += piece;
    }
    return out;
}

bool BigInt::fits_ll() const {
    if (limbs_.size() > 3) return false;
    const std::string text = str();
    try {
        std::size_t used = 0;
        (void)std::stoll(text, &used);
        return used == text.size();
    } catch (...) {
        return false;
    }
}

long long BigInt::to_ll() const { return std::stoll(str()); }

}  // namespace uq
