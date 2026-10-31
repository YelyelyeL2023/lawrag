#!/bin/sh
# Keep files over GitHub's 100 MiB limit out of the repo: append each one to
# .gitignore and drop it from the index (the file on disk is left alone).
#
#   sh scripts/ignore_large_files.sh            scan the whole working tree
#   sh scripts/ignore_large_files.sh --staged   scan only staged files (pre-commit hook)
#   sh scripts/ignore_large_files.sh --install  install as .git/hooks/pre-commit
#
# LIMIT_BYTES overrides the threshold (default 104857600 = 100 MiB).

set -eu
cd "$(git rev-parse --show-toplevel)"

LIMIT=${LIMIT_BYTES:-104857600}
mode=${1:-}

if [ "$mode" = "--install" ]; then
    hook="$(git rev-parse --git-path hooks)/pre-commit"
    printf '#!/bin/sh\nexec sh scripts/ignore_large_files.sh --staged\n' > "$hook"
    chmod +x "$hook"
    echo "installed $hook"
    exit 0
fi

list=$(mktemp)
trap 'rm -f "$list"' EXIT

if [ "$mode" = "--staged" ]; then
    # Size of the staged blob, not the working copy.
    git -c core.quotePath=false diff --cached --name-only --diff-filter=ACMR | while IFS= read -r f; do
        s=$(git cat-file -s ":$f" 2>/dev/null) || continue
        if [ "$s" -gt "$LIMIT" ]; then printf '%s\t%s\n' "$s" "$f"; fi
    done > "$list"
else
    git -c core.quotePath=false ls-files --cached --others --exclude-standard | while IFS= read -r f; do
        [ -f "$f" ] || continue
        s=$(wc -c < "$f")
        if [ "$s" -gt "$LIMIT" ]; then printf '%s\t%s\n' "$s" "$f"; fi
    done > "$list"
fi

[ -s "$list" ] || exit 0

touch .gitignore
if ! grep -qx '# Files over the GitHub size limit (added by scripts/ignore_large_files.sh)' .gitignore; then
    [ -n "$(tail -c1 .gitignore)" ] && echo >> .gitignore
    echo '# Files over the GitHub size limit (added by scripts/ignore_large_files.sh)' >> .gitignore
fi

while IFS="$(printf '\t')" read -r s f; do
    [ "$f" = .gitignore ] && continue
    # Anchor to the repo root and escape glob characters so the rule matches only this file.
    rule="/$(printf '%s' "$f" | sed 's/[][*?\\]/\\&/g')"
    grep -qxF "$rule" .gitignore || echo "$rule" >> .gitignore
    git rm --cached --quiet --ignore-unmatch -- "$f"
    echo "ignored: $f ($((s / 1048576)) MiB)" >&2
done < "$list"

git add .gitignore
