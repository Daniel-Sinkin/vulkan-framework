#pragma once

#include <cassert>
#include <concepts>
#include <utility>

namespace ds_vk
{
template <typename T, typename U, typename V>
    requires requires(T x, U lo, V hi) {
        std::cmp_less_equal(lo, x);
        std::cmp_less_equal(x, hi);
        std::cmp_less_equal(lo, hi);
    }
[[nodiscard]] constexpr auto in_interval(T x, U lo, V hi) -> bool
{
    assert(std::cmp_less_equal(lo, hi));
    return std::cmp_less_equal(lo, x) && std::cmp_less_equal(x, hi);
}

template <std::floating_point T>
[[nodiscard]] constexpr auto in_interval(T x, T lo, T hi) -> bool
{
    assert(lo <= hi);
    return lo <= x && x <= hi;
}
}  // namespace ds_vk
