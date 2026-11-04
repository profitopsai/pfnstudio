#!/usr/bin/env bash
#
# Operator interface for the pfnstudio OSS repo. One consistent entrypoint
# for the release chores so nobody has to remember the build/version/twine
# dance. Two publishable packages:
#
#   cli  -> packages/cli   -> pfnstudio        -> tag cli-v<version>
#   core -> packages/core  -> pfnstudio-core   -> tag core-v<version>
#
# Usage:
#   ./run.sh version                 show current cli + core versions
#   ./run.sh build   <cli|core|all>  build wheel + sdist (uv build)
#   ./run.sh publish <cli|core>      build + upload to PyPI (twine)
#   ./run.sh tag     <cli|core>      git tag <prefix><version> + push (CI publish)
#   ./run.sh verify  [slug...]       run every open model from a fresh venv
#
# Twine resolution honors $VENV/twine if set (existing operator flow),
# else a `twine` on PATH, else `uv run --with twine twine`. PyPI creds
# come from ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD, as usual.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

die() { echo "✗ $*" >&2; exit 1; }

pkg_dir() {
  case "$1" in
    cli|pfnstudio)       echo "packages/cli" ;;
    core|pfnstudio-core) echo "packages/core" ;;
    *) die "unknown package '$1' (expected: cli | core)" ;;
  esac
}

pkg_name() {
  case "$1" in
    cli|pfnstudio)       echo "pfnstudio" ;;
    core|pfnstudio-core) echo "pfnstudio-core" ;;
    *) die "unknown package '$1' (expected: cli | core)" ;;
  esac
}

tag_prefix() {
  case "$1" in
    cli|pfnstudio)       echo "cli-v" ;;
    core|pfnstudio-core) echo "core-v" ;;
    *) die "unknown package '$1' (expected: cli | core)" ;;
  esac
}

pkg_version() {
  grep -E '^version' "$(pkg_dir "$1")/pyproject.toml" | head -1 | sed -E 's/.*"([^"]+)".*/\1/'
}

twine_cmd() {
  if [[ -n "${VENV:-}" && -x "$VENV/twine" ]]; then echo "$VENV/twine"; return; fi
  if command -v twine >/dev/null 2>&1; then echo "twine"; return; fi
  echo "uv run --with twine twine"
}

# 0 (true) if this exact version is already on PyPI.
version_on_pypi() {
  local name="$1" ver="$2"
  curl -fsS "https://pypi.org/pypi/$name/json" 2>/dev/null \
    | python3 -c "import sys,json;sys.exit(0 if '$ver' in json.load(sys.stdin).get('releases',{}) else 1)" \
    2>/dev/null
}

cmd_version() {
  printf 'pfnstudio (cli):  %s\n' "$(pkg_version cli)"
  printf 'pfnstudio-core:   %s\n' "$(pkg_version core)"
}

cmd_build() {
  local pkg="${1:-all}"
  if [[ "$pkg" == all ]]; then cmd_build cli; cmd_build core; return; fi
  local dir; dir="$(pkg_dir "$pkg")"
  echo "→ building $(pkg_name "$pkg") $(pkg_version "$pkg")"
  ( cd "$dir" && rm -rf dist && uv build )
  ls -1 "$dir/dist"
}

cmd_publish() {
  local pkg="${1:?usage: ./run.sh publish <cli|core>}"
  local dir name ver tw
  dir="$(pkg_dir "$pkg")"; name="$(pkg_name "$pkg")"; ver="$(pkg_version "$pkg")"
  if version_on_pypi "$name" "$ver"; then
    die "$name $ver is already on PyPI — bump the version in $dir/pyproject.toml first."
  fi
  cmd_build "$pkg"           # always build fresh so dist matches pyproject
  tw="$(twine_cmd)"
  echo "→ uploading $name $ver to PyPI via: $tw"
  $tw upload "$dir/dist/"*
  echo "✓ published $name $ver  ·  verify: pip install -U $name==$ver"
}

cmd_tag() {
  local pkg="${1:?usage: ./run.sh tag <cli|core>}"
  local ver tag
  ver="$(pkg_version "$pkg")"; tag="$(tag_prefix "$pkg")${ver}"
  echo "→ tagging $tag (push triggers CI Trusted-Publishing)"
  git tag "$tag"
  git push origin "$tag"
  echo "✓ pushed $tag"
}

usage() {
  cat <<'EOF'
pfnstudio operator interface — release ops for the two packages.

  ./run.sh version                 show current cli + core versions
  ./run.sh build   <cli|core|all>  build wheel + sdist (uv build)
  ./run.sh publish <cli|core>      build + upload to PyPI (twine)
  ./run.sh tag     <cli|core>      git tag <prefix><version> + push (CI publish)
  ./run.sh verify  [slug...]       before publishing: build this checkout into a
                                   fresh venv and run each open model, cold, on
                                   its sample and with --input (scripts/verify-models.sh)

Packages:  cli = pfnstudio   ·   core = pfnstudio-core
Twine:     honors $VENV/twine if set, else `twine`, else `uv run --with twine`.
PyPI creds via ~/.pypirc or TWINE_USERNAME/TWINE_PASSWORD.
EOF
}

case "${1:-}" in
  version)          cmd_version ;;
  build)   shift;   cmd_build "${1:-all}" ;;
  publish) shift;   cmd_publish "$@" ;;
  tag)     shift;   cmd_tag "$@" ;;
  verify)  shift;   exec "$ROOT/scripts/verify-models.sh" "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown command '$1'" >&2; usage; exit 1 ;;
esac
