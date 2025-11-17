#!/usr/bin/env bash

set -euo pipefail

echo "Building..."
g++ -std=c++17 -O2 -pthread main.cpp -o main.out

echo
echo "=== Test 1: Queue latency vs load (shorter per) ==="
PER=200000
CAP=65536
./main.out --test perf_queue --producers 1 --consumers 1 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 1 --consumers 2 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 1 --consumers 4 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 1 --consumers 8 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 2 --consumers 1 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 4 --consumers 1 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 8 --consumers 1 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 2 --consumers 2 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 4 --consumers 4 --per ${PER} --capacity ${CAP}
./main.out --test perf_queue --producers 8 --consumers 8 --per ${PER} --capacity ${CAP}

echo
echo "=== Test 2: Queue latency vs per (scale per, fixed 4p x 4c) ==="
for PER2 in 50000 100000 200000 400000; do
  ./main.out --test perf_queue --producers 4 --consumers 4 --per ${PER2} --capacity ${CAP}
done

echo
echo "=== Test 3: Queue vs List (list uses pop_front removal => faster) ==="
./main.out --test perf_queue --producers 4 --consumers 4 --per ${PER} --capacity ${CAP}
# For list we keep PER but pop_front makes removals O(1)
./main.out --test perf_list --producers 4 --per ${PER}

echo
echo "=== Test 4: Queue capacity sweep (use moderate per) ==="
for CAP2 in 1024 4096 16384 65536; do
  ./main.out --test perf_queue --producers 4 --consumers 4 --per ${PER} --capacity ${CAP2}
done

echo
echo "=== Test 5: List throughput vs producers (pop_front removal) ==="
PERL=200000
./main.out --test perf_list --producers 1 --per ${PERL}
./main.out --test perf_list --producers 2 --per ${PERL}
./main.out --test perf_list --producers 4 --per ${PERL}
./main.out --test perf_list --producers 8 --per ${PERL}

echo
echo "=== Test 6: List latency vs per (4 producers) ==="
for PER3 in 50000 100000 200000 400000; do
  ./main.out --test perf_list --producers 4 --per ${PER3}
done

echo
echo "All tests finished. Results appended to perf_results.csv. Use plot_perf.py to generate plots."
