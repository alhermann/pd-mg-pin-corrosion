#pragma once
#include <array>
#include <cmath>
#include <chrono>
#include <string>
#include <cstdio>

#ifndef PD_DIM
#define PD_DIM 2
#endif

constexpr int DIM = PD_DIM;

using Vec = std::array<double, DIM>;

inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (int d = 0; d < DIM; ++d) s += a[d] * b[d];
    return s;
}

inline double norm(const Vec& a) {
    return std::sqrt(dot(a, a));
}

inline Vec operator+(const Vec& a, const Vec& b) {
    Vec r;
    for (int d = 0; d < DIM; ++d) r[d] = a[d] + b[d];
    return r;
}

inline Vec operator-(const Vec& a, const Vec& b) {
    Vec r;
    for (int d = 0; d < DIM; ++d) r[d] = a[d] - b[d];
    return r;
}

inline Vec operator*(double s, const Vec& a) {
    Vec r;
    for (int d = 0; d < DIM; ++d) r[d] = s * a[d];
    return r;
}

inline Vec operator*(const Vec& a, double s) {
    return s * a;
}

inline Vec operator/(const Vec& a, double s) {
    Vec r;
    double inv = 1.0 / s;
    for (int d = 0; d < DIM; ++d) r[d] = a[d] * inv;
    return r;
}

inline Vec& operator+=(Vec& a, const Vec& b) {
    for (int d = 0; d < DIM; ++d) a[d] += b[d];
    return a;
}

inline Vec& operator-=(Vec& a, const Vec& b) {
    for (int d = 0; d < DIM; ++d) a[d] -= b[d];
    return a;
}

inline Vec vec_zero() {
    Vec v{};
    return v;
}

// Construct a position Vec from up to 3 coordinates (extra coords ignored for 2D)
inline Vec make_vec(double x, double y, double z = 0.0) {
    Vec v{};
    v[0] = x;
    v[1] = y;
    if constexpr (DIM >= 3) v[2] = z;
    return v;
}

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;
    std::string label_;

    Timer(const std::string& label) : label_(label), start_(Clock::now()) {}

    double elapsed_s() const {
        auto now = Clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    void report() const {
        std::printf("  [Timer] %s: %.3f s\n", label_.c_str(), elapsed_s());
    }

    void reset() { start_ = Clock::now(); }
};

// Constants
constexpr double PI = 3.14159265358979323846;
