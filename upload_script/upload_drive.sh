#!/bin/bash
set -euo pipefail

# ---------------- CONFIG ----------------
REMOTE="crg:DATA"   # your rclone remote
SOURCE="${1:-}"
DRY_RUN="${2:-}"    # optional --dry-run
# ---------------------------------------

if [[ -z "$SOURCE" ]]; then
    echo "Usage: $0 <source folder> [--dry-run]"
    exit 1
fi

echo "Scanning for game files in: $SOURCE"

# Pre-create game/subfolders
for GAME in A B C; do
  for SUB in "Audio" "Eye Tracking" "Game States"; do
    if [[ -n "$DRY_RUN" ]]; then
      echo "Would create $REMOTE/Game $GAME/$SUB"
    else
      rclone mkdir "$REMOTE/Game $GAME/$SUB"
    fi
  done
done

# Collect all relevant files
FILES=()
while IFS= read -r f; do
  FILES+=("$f")
done < <(find "$SOURCE" -type f -name "puzzle-game*-*.*")

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files found!"
    exit 0
fi

# Generate participant code from first file's timestamp
FIRST_FILE="${FILES[0]}"
TIMECODE=$(echo "$FIRST_FILE" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}')
PARTICIPANT_CODE=$(echo -n "$TIMECODE" | md5 | tr -dc '0-9' | head -c10)
echo "Participant code for this run: $PARTICIPANT_CODE"

# Collision-safe naming
make_unique_name() {
  local dest="$1"
  local base="${2:-unknown}"
  local ext="${3:-txt}"
  local counter=0
  local newname="$base.$ext"

  while true; do
    if rclone lsf --files-only "$dest" 2>/dev/null | grep -qx "$newname"; then
      ((counter++))
      newname="${base}_$counter.$ext"
    else
      echo "$newname"
      return
    fi
  done
}

# Process each file
for FILE in "${FILES[@]}"; do
  BASENAME=$(basename "$FILE")
  EXT="${BASENAME##*.}"
  NAME_NO_EXT="${BASENAME%.*}"

  if [[ -z "$NAME_NO_EXT" || -z "$EXT" ]]; then
      echo "Skipping invalid file: $FILE"
      continue
  fi

  # Determine subfolder
  if [[ "$NAME_NO_EXT" == *"-audio"* ]]; then
    SUBFOLDER="Audio"
  elif [[ "$NAME_NO_EXT" == *"eye-tracking"* ]]; then
    SUBFOLDER="Eye Tracking"
  elif [[ "$NAME_NO_EXT" == *"-state"* ]]; then
    SUBFOLDER="Game States"
  else
    echo "Skipping unknown file type: $FILE"
    continue
  fi

  # Determine game letter
  GAME_NUM=$(echo "$NAME_NO_EXT" | sed -E 's/.*game([1-3]).*/\1/')
  case "$GAME_NUM" in
    1) GAME="A" ;;
    2) GAME="B" ;;
    3) GAME="C" ;;
    *) echo "Unknown game in file $FILE"; continue ;;
  esac

  # Replace full timestamp (including milliseconds/Z) with participant code
  NEW_BASE=$(echo "$NAME_NO_EXT" | sed -E 's/(game[1-3]-.*-)[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}.*$/\1'"$PARTICIPANT_CODE"'/')

  DEST="$REMOTE/Game $GAME/$SUBFOLDER"
  NEW_NAME=$(make_unique_name "$DEST" "$NEW_BASE" "$EXT")

  echo "Copying:"
  echo "  $FILE"
  echo "  → Game $GAME / $SUBFOLDER / $NEW_NAME"

  if [[ -n "$DRY_RUN" ]]; then
    rclone copyto --dry-run "$FILE" "$DEST/$NEW_NAME"
  else
    rclone copyto "$FILE" "$DEST/$NEW_NAME"
  fi
done

echo "Done!"

