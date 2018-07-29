#include <cassert>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include <x86intrin.h>

#include "timer.hh"

static const uint64_t MASK = 0x808080808080;
static const uint64_t XMASK = 0x7f7f7f7f7f7f;

inline uint64_t readFastUInt1(uint64_t x)
{
  uint64_t m = XMASK & _blsmsk_u64(x & MASK);
  return _pext_u64(x, m);
}

inline uint64_t readFastUInt2(uint64_t x)
{
  static constexpr const uint64_t masks[] = {
    0x7f,
    0x7f7f,
    0x7f7f7f,
    0x7f7f7f7f,
    0x7f7f7f7f7f,
    0x7f7f7f7f7f7f,
  };
  unsigned size = _tzcnt_u64(x & MASK) + 1;
  size >>= 3;
  uint64_t m = masks[size - 1];
  return _pext_u64(x, m);
}

uint64_t encode(uint64_t x)
{
  uint64_t r = _pdep_u64(x, XMASK);
  uint64_t m = 1ull << 7;
  for (uint64_t i = r >> 8; i != 0; i >>= 8) {
    m <<= 8;
  }
  return r | m;
}

std::vector<uint64_t> make_data(size_t s)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1, 1 << 25);
  std::vector<uint64_t> orig;
  for (uint64_t x = 1; x <= s; ++x) {
    orig.push_back(encode(dis(gen)));
  }
  return orig;
}

int main()
{
  auto data = make_data(1 << 26);

  double t0;
  { scoped_timer clock {t0};
    uint64_t y;
    for (auto x : data) {
      y = readFastUInt1(x);
      if (y == 0)
        break;
    }
  }

  double t1;
  { scoped_timer clock {t1};
    uint64_t y;
    for (auto x : data) {
      y = readFastUInt2(x);
      if (y == 0)
        break;
    }
  }
  std::cout << t0 << " " << t1 << std::endl;
  return 0;
}
