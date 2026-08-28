#!/usr/bin/env bash
# Assemble the `userdocs` domain: documentation for the same software the man pages
# describe, written for end users rather than as manual pages.
#
# This is the *adjacent* domain in phase 5. The literature domain is unrelated enough
# that it never competes for a retrieval slot, which makes it a test that cannot
# fail; package READMEs and GNOME help talk about the same tools in the same
# vocabulary, which is where dilution actually shows up.
#
#   ./scripts/collect_userdocs.sh && .venv/bin/python scripts/ingest.py \
#       --db data/index/phase5.db --domain userdocs data/corpus/domains/userdocs/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/corpus/domains/userdocs"
rm -rf "$OUT"; mkdir -p "$OUT"

# Package documentation. Changelogs and copyright files are excluded: they are
# near-identical across hundreds of packages and would be the most duplicated
# passage in the index. Everything is normalised to .txt so the ingest collector
# picks it up regardless of the original suffix (.Debian, .rst, no suffix at all).
find /usr/share/doc -maxdepth 3 \( -name 'README*' -o -name '*.md' -o -name '*.txt' \) \
     ! -name '*.gz' ! -iname 'copyright*' ! -iname 'changelog*' \
     -size +2k -size -400k 2>/dev/null | head -900 | while read -r f; do
  pkg=$(echo "$f" | sed 's|/usr/share/doc/||; s|/.*||')
  cp "$f" "$OUT/pkg-${pkg}-$(basename "$f" | tr -c 'A-Za-z0-9._-' '_').txt" 2>/dev/null || true
done

# GNOME end-user help. Mallard is XML, so tags come off and the prose stays; only
# the /C/ (English) tree is taken, since the corpus is English-only.
find /usr/share/help -path '*/C/*' -name '*.page' 2>/dev/null | head -1000 | while read -r f; do
  app=$(echo "$f" | sed 's|/usr/share/help/C/||; s|/.*||')
  sed -e 's/<[^>]*>/ /g' -e 's/&[a-z]*;/ /g' "$f" | tr -s ' \n' ' \n' \
    > "$OUT/help-${app}-$(basename "$f" .page).txt" 2>/dev/null || true
done

echo "$(ls "$OUT" | wc -l) files, $(du -sh "$OUT" | cut -f1) -> $OUT"
