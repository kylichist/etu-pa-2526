#!/bin/bash

rm -f main.out
g++ -O3 -march=native -std=c++17 -pthread main.cpp -o main.out
./main.out
