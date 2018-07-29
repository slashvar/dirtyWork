#include <cassert>
#include <cstdio>
#include <deque>
#include <iostream>
#include <random>
#include <string.h>
#include <vector>
#include <x86intrin.h>

#include "timer.hh"

static const uint64_t MASK = 0x808080808080;
static const uint64_t XMASK = 0x7f7f7f7f7f7f;

/*
 * Use AVX2/BMI2 instruction PEXT
 */
inline uint64_t readFastUInt(uint8_t buffer[], uint64_t& position)
{
  static constexpr const uint64_t masks[] = {
    0x7f,
    0x7f7f,
    0x7f7f7f,
    0x7f7f7f7f,
    0x7f7f7f7f7f,
    0x7f7f7f7f7f7f,
  };
  uint64_t x;
  memcpy(&x, buffer + position, 8);
  unsigned size = _tzcnt_u64(x & MASK) + 1;
  size >>= 3;
  uint64_t m = XMASK & _blsmsk_u64(x & MASK);
  position += size;
  return _pext_u64(x, m);
}

struct prefetch_reader
{
  prefetch_reader(uint8_t buffer_[], uint64_t& pos)
    : buffer(buffer_), position(pos), pend(position + psize), pbeg(position)
  {
    memcpy(prefetch, buffer + position, psize);
  }

  uint64_t read()
  {
    if (pend - position < sizeof (uint64_t)) {
      memcpy(prefetch, buffer + position, psize);
      pbeg = position;
      pend = position + psize;
    }
    __builtin_prefetch(prefetch + (position - pbeg), 0, 1);
    uint64_t x = *reinterpret_cast<uint64_t*>(prefetch + (position - pbeg));
    unsigned size = _tzcnt_u64(x & MASK) + 1;
    size >>= 3;
    uint64_t m = XMASK & _blsmsk_u64(x & MASK);
    position += size;
    return _pext_u64(x, m);
  }

  static const uint64_t psize = 32 * sizeof (uint64_t);
  uint8_t prefetch[psize];
  uint8_t* buffer;
  uint64_t& position;
  uint64_t pend;
  uint64_t pbeg;
  uint64_t cur;
};

inline void skipFastUInt(uint8_t buffer[], uint64_t& position)
{
  uint64_t x;
  memcpy(&x, buffer + position, 8);
  position += (_tzcnt_u64(x & MASK) + 1) >> 3;
}

inline void writeFastUInt(uint64_t val, std::vector<uint8_t>& output)
{
  if (val < (1U << 7)) {
    output.push_back(val | (1U << 7));
  } else if (val < (1U << 14)) {
    output.push_back(val & 127);
    output.push_back((val >> 7) | (1U << 7));
  } else if (val < (1U << 21)) {
    output.push_back(val & 127);
    output.push_back((val >> 7) & 127);
    output.push_back((val >> 14) | (1U << 7));
  } else if (val < (1U << 28)) {
    output.push_back(val & 127);
    output.push_back((val >> 7) & 127);
    output.push_back((val >> 14) & 127);
    output.push_back((val >> 21) | (1U << 7));
  } else if (val < (1ULL << 35)) {
    output.push_back(val & 127);
    output.push_back((val >> 7) & 127);
    output.push_back((val >> 14) & 127);
    output.push_back((val >> 21) & 127);
    output.push_back((val >> 28) | (1U << 7));
  } else {
    output.push_back(val & 127);
    output.push_back((val >> 7) & 127);
    output.push_back((val >> 14) & 127);
    output.push_back((val >> 21) & 127);
    output.push_back((val >> 28) & 127);
    output.push_back((val >> 35) | (1U << 7));
  }
}

template<typename Vect>
std::vector<uint8_t> write(const Vect& orig)
{
  std::vector<uint8_t> out;
  for (auto x : orig) {
    writeFastUInt(x, out);
  }
  writeFastUInt(0, out);
  return out;
}

std::vector<uint8_t> dont_keep_orig(size_t s)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> dis(1, 1 << 20);
  std::vector<uint64_t> orig;
  for (uint64_t x = 1; x <= s; ++x) {
    orig.push_back(dis(gen));
  }
  return write(orig);
}

int main()
{
  printf("MASK: %lx\n", MASK);
  std::cout << "Generate data ...\n";
  auto data = dont_keep_orig(1 << 26);
  std::cout << "... warmin up skip content ...\n";
  uint64_t position = 0;
  while (position < data.size()) {
    skipFastUInt(data.data(), position);
  }
  std::cout << position << std::endl;
  position = 0;
  double t;
  { scoped_timer timer(t);
    while (position < data.size()) {
      skipFastUInt(data.data(), position);
    }
  }
  std::cout << position << " " << t << std::endl;
  t = 0;
  position = 0;
  { scoped_timer timer(t);
    while (position < data.size()) {
      skipFastUInt(data.data(), position);
    }
  }
  std::cout << position << " " << t << std::endl;
  t = 0;
  position = 0;
  { scoped_timer timer(t);
    while (position < data.size()) {
      uint64_t x = readFastUInt(data.data(), position);
      if (x == 0) break;
    }
  }
  std::cout << position << " " << t << std::endl;

  t = 0;
  position = 0;
  { scoped_timer timer(t);
    prefetch_reader reader(data.data(), position);
    while (position < data.size()) {
      uint64_t x = reader.read();
      if (x == 0) break;
    }
  }
  std::cout << position << " " << t << std::endl;
  return 0;
}
