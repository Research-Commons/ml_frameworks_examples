#!/bin/bash
set -e

# Run the main CNN program
./main

# Clean up after execution
rm -f train.csv test.csv