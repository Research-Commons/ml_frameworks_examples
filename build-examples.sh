#!/bin/bash

set -e

# ------------------------
# Colors
# ------------------------
GREEN="\033[1;32m"
RED="\033[1;31m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RESET="\033[0m"

# ------------------------
# ASCII Header
# ------------------------
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║              Build Script for Examples               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ------------------------
# Usecase Mappings
# ------------------------
declare -A USECASE_MAP=(
  [1]="Tabular-Regression–MLP"
  [2]="Titanic-Survival-Prediction-XGBoost"
)

# ------------------------
# Help Message
# ------------------------
usage() {
  echo -e "${YELLOW}Usage:${RESET} $0 --framework <libtorch|pytorch> --usecase <number> [--rebuild] [--copy-resources <path>] [--dir <base|exp>]"
  echo -e "\n${YELLOW}Available Usecases:${RESET}"
  for key in "${!USECASE_MAP[@]}"; do
    echo "  $key => ${USECASE_MAP[$key]}"
  done
  exit 1
}

# ------------------------
# Parse Arguments
# ------------------------
FRAMEWORK=""
USECASE_ID=""
REBUILD=false
RESOURCE_PATH=""
DIR_MODE="exp"  # default directory is 'experimental'

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --framework) FRAMEWORK="$2"; shift ;;
    --usecase)   USECASE_ID="$2"; shift ;;
    --dir)
      if [[ "$2" != "base" && "$2" != "exp" ]]; then
        echo -e "${RED}[!] Invalid value for --dir: '$2'. Use 'base' or 'exp'.${RESET}"
        usage
      fi
      DIR_MODE="$2"
      shift
      ;;
    --rebuild)   REBUILD=true ;;
    --copy-resources) RESOURCE_PATH="$2"; shift ;;
    *) echo -e "${RED}Unknown option: $1${RESET}"; usage ;;
  esac
  shift
done

# ------------------------
# Validate and Resolve
# ------------------------
if [[ -z "$FRAMEWORK" || -z "$USECASE_ID" ]]; then
  usage
fi

FRAMEWORK=$(echo "$FRAMEWORK" | tr '[:upper:]' '[:lower:]')
USECASE_NAME="${USECASE_MAP[$USECASE_ID]}"

if [[ -z "$USECASE_NAME" ]]; then
  echo -e "${RED}[!] Invalid usecase ID: $USECASE_ID${RESET}"
  usage
fi

EXAMPLE_DIR="examples/$USECASE_NAME/$FRAMEWORK/$DIR_MODE"
IMAGE_TAG="${FRAMEWORK}-$(echo "$USECASE_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-')"
DOCKERFILE="$EXAMPLE_DIR/Dockerfile"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo -e "${RED}[!] Dockerfile not found: $DOCKERFILE${RESET}"
  exit 1
fi

# ------------------------
# Optional Resource Copy
# ------------------------
if [[ -n "$RESOURCE_PATH" ]]; then
  if [[ ! -e "$RESOURCE_PATH" ]]; then
    echo -e "${RED}[!] Resource path not found: $RESOURCE_PATH${RESET}"
    exit 1
  fi

  echo -e "${GREEN}[*] Copying resources from '${RESOURCE_PATH}' → '${EXAMPLE_DIR}/resources/'${RESET}"
  mkdir -p "$EXAMPLE_DIR/resources/"
  cp -r "$RESOURCE_PATH"/* "$EXAMPLE_DIR/resources/"
fi

# ------------------------
# Build Docker Image
# ------------------------
if $REBUILD || [[ "$(docker images -q $IMAGE_TAG 2> /dev/null)" == "" ]]; then
  echo -e "${GREEN}[*] Building Docker image: ${IMAGE_TAG}${RESET}"
  docker build -t "$IMAGE_TAG" "$EXAMPLE_DIR"
else
  echo -e "${YELLOW}[!] Skipping build: image already exists. Use --rebuild to force rebuild.${RESET}"
fi

# ------------------------
# Final Summary
# ------------------------
echo -e "\n${BLUE}╭─────────────────────────────╮"
echo    "│      ✅ Build Complete      │ "
echo -e "╰─────────────────────────────╯${RESET}"

echo -e "${GREEN}[*] Docker image: ${IMAGE_TAG}${RESET}"
echo -e "${GREEN}[*] Usecase:      ${USECASE_NAME}${RESET}"
echo -e "${GREEN}[*] Framework:    ${FRAMEWORK}${RESET}"

echo -e "\n${YELLOW}👉 To run the container:${RESET}"
echo -e "${BLUE}   docker run --cpus=\"2\" --memory=\"4g\" --memory-swap=\"4g\" -it ${IMAGE_TAG} <cmd-line-args>${RESET}\n"
