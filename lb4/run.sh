#!/bin/bash

nvcc -x cu main.cpp -O3 -std=c++17 -o main.out
./main.out
