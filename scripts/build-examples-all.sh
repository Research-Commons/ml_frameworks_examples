#!/bin/bash

set -euo pipefail

GREEN="\033[1;32m"
BLUE="\033[1;34m"
RESET="\033[0m"

echo -e "${BLUE}▶ Running all Docker example builds...${RESET}"

for fw in libtorch pytorch; do
  for uc in 1; do
    echo -e "${GREEN}→ Building: framework=${fw}, usecase=${uc}${RESET}"
    bash build-examples.sh --framework "$fw" --usecase "$uc" --rebuild
  done
done

echo -e "${BLUE}✅ All builds finished.${RESET}"
