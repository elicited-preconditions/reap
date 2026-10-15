#!/usr/bin/env bash
# Download the REAP v1.0 release assets, check their sha256 and extract them at the repository root.
#
#   bash scripts/download_assets.sh                      every group
#   bash scripts/download_assets.sh tabletop datasets    only these groups
#
# Groups: tabletop confined multilevel_shelf stacking access19 baselines (checkpoints/<group>/),
# datasets (data/<env>/).
# The tarballs are kept in assets/. A tarball that is already there and matches its sha256 is not downloaded again.
# REAP_ASSET_URL replaces the download location (default: the GitHub release v1.0 of the repository).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
URL=${REAP_ASSET_URL:-https://github.com/elicited-preconditions/reap/releases/download/v1.0}
DIR=$ROOT/assets

# identical to SHA256SUMS of the release
SUMS="c09c6a0670dfcf40fb0dacbc451faf90035981a6caf76483508ac05b6c96167e  reap_ckpt_tabletop.tar.gz
4462416a5298ce7d92821018c907721da1bbacb22fd392f67d994fa1ca0f3508  reap_ckpt_confined.tar.gz
101efd3573c6bac7184226c411be7aea1c36ced7acb6749c1f3bd2e474203295  reap_ckpt_multilevel_shelf.tar.gz
4abf729e8b2c7629fc9f7d7102679c1ca89d93436e18980bcd8fd60eb4ecb380  reap_ckpt_stacking.tar.gz
6978b23cfae44bac0b64cb51f95ea5cdbee0f4be754eb5f108cb7afeda062dbe  reap_ckpt_access19.tar.gz
a65c4757e14fe38f7a0861d1db472c723b7b966baac8d3240ad41d6b0a350446  reap_ckpt_baselines.tar.gz
7fecdfdedb883207fd5fe0d9b533f3f111fe231ec5afff17684403ea40af0e90  reap_datasets.tar.gz"

die() { echo "download_assets.sh: $*" >&2; exit 1; }

sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
    else die "needs sha256sum or shasum"; fi
}

fetch() {  # fetch <url> <file>
    if command -v curl >/dev/null; then curl -fL --retry 3 --progress-bar -o "$2" "$1"
    elif command -v wget >/dev/null; then wget -O "$2" "$1"
    else die "needs curl or wget"; fi
}

file_of() {  # group name -> tarball name
    case $1 in
        datasets) echo reap_datasets.tar.gz ;;
        *) echo "reap_ckpt_$1.tar.gz" ;;
    esac
}

if [ $# -eq 0 ]; then
    set -- tabletop confined multilevel_shelf stacking access19 baselines datasets
fi
mkdir -p "$DIR"
for group in "$@"; do
    f=$(file_of "$group")
    want=$(printf '%s\n' "$SUMS" | awk -v f="$f" '$2 == f {print $1}')
    [ -n "$want" ] || die "unknown group '$group'"
    if [ -f "$DIR/$f" ] && [ "$(sha256 "$DIR/$f")" = "$want" ]; then
        echo "$f: present, sha256 ok"
    else
        echo "$f: downloading $URL/$f"
        fetch "$URL/$f" "$DIR/$f.part" || { rm -f "$DIR/$f.part"; die "download of $f failed"; }
        got=$(sha256 "$DIR/$f.part")
        [ "$got" = "$want" ] || { rm -f "$DIR/$f.part"; die "$f: sha256 $got, expected $want"; }
        mv "$DIR/$f.part" "$DIR/$f"
    fi
    # every member must lie under checkpoints/ or data/ (no ..) and must not be a file tracked in git
    members=$(tar -tzf "$DIR/$f")
    bad=$(printf '%s\n' "$members" | awk '$0 != "" && ($0 !~ /^(checkpoints|data)\// || $0 ~ /(^|\/)\.\.(\/|$)/)')
    [ -z "$bad" ] || die "$f: members outside checkpoints/ and data/: $(echo "$bad" | head -3)"
    if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        tracked=$(LC_ALL=C comm -12 <(printf '%s\n' "$members" | LC_ALL=C sort) <(git -C "$ROOT" ls-files | LC_ALL=C sort))
        [ -z "$tracked" ] || die "$f: would overwrite files tracked in git: $(echo "$tracked" | head -3)"
    fi
    tar -xzf "$DIR/$f" -C "$ROOT"
    echo "$f: extracted into $ROOT"
done
