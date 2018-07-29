# Makefile

CXX = clang++
CPPFLAGS =
CXXFLAGS = -g -Wall -Wextra -std=c++14 -Ofast -mbmi2 -march=native -mtune=native

all:

foo-prof: foo.cc timer.hh
	${CXX} ${CXXFLAGS} -fprofile-instr-generate -o $@ foo.cc

foo.profdata: foo-prof
	LLVM_PROFILE_FILE="foo-%p.profraw" ./foo-prof
	LLVM_PROFILE_FILE="foo-%p.profraw" ./foo-prof
	LLVM_PROFILE_FILE="foo-%p.profraw" ./foo-prof
	LLVM_PROFILE_FILE="foo-%p.profraw" ./foo-prof
	llvm-profdata merge -output=$@ foo-*.profraw

foo-opt: foo.cc foo.profdata
	${CXX} ${CXXFLAGS} -fprofile-instr-use=foo.profdata -o $@ foo.cc

clean:
	${RM} *.o *.profraw
