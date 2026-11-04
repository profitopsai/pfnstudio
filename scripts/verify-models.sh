#!/usr/bin/env bash
#
# Pre-publish check: install THIS checkout's pfnstudio + pfnstudio-core into a
# fresh venv and run every open base model the way the website tells a user
# to — before anything reaches PyPI.
#
#   ./run.sh verify                      every open model in the marketplace
#   ./run.sh verify ceptron-xs-discovery cdfm-base
#
# Per model, from a cold start (empty venv: no torch, no adapter, no weights):
#   1. pfnstudio run <slug> --install-deps                    its own sample
#   2. pfnstudio run <slug> --install-deps --input input.json input.json built
#      from the sample the running model declares (warm: reuses 1's cache)
#   3. with VERIFY_SITE set (e.g. https://pfnstudio.com): the site's
#      /models/<slug>/input.json must equal the file from 2
# A case passes only when the command exits 0 AND prints a result.
#
# Env: PYTHON (default python3, needs 3.10+), PFNSTUDIO_API_URL (marketplace,
# default hosted), VERIFY_SITE, VERIFY_DIR (work dir, default a new temp dir;
# kept afterwards for the logs), VERIFY_PORT (daemon port, default 11541 so a
# daemon already on 11440 is never touched).
#
# Needs: RAM for the largest model plus PyTorch; disk for torch + every
# model's packages and weights. Models are unloaded one by one as they finish.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
API="${PFNSTUDIO_API_URL:-https://cloud.pfnstudio.com/api}"
SITE="${VERIFY_SITE:-}"
PORT="${VERIFY_PORT:-11541}"
WORK="${VERIFY_DIR:-$(mktemp -d -t pfnstudio-verify.XXXXXX)}"
mkdir -p "$WORK/logs" "$WORK/inputs"
VENV="$WORK/venv"
DAEMON="http://127.0.0.1:$PORT"

# Everything the CLI caches lives under $WORK: nothing from an earlier install
# on this machine is reused, so the first run of each model is a real cold one.
export PFNSTUDIO_API_URL="$API"
export PFNSTUDIO_HOST="$DAEMON"
export PFNSTUDIO_BASEMODELS_CACHE="$WORK/cache/basemodels"
export PFNSTUDIO_CHECKPOINT_CACHE="$WORK/cache/checkpoints"
export PFNSTUDIO_DAEMON_LOG="$WORK/daemon.log"
export HF_HOME="$WORK/cache/hf"
unset PFNSTUDIO_BASEMODELS_DIR PFNSTUDIO_SERVE_TOKEN

stop_daemon() { pkill -f "pfnstudio.daemon --host 127.0.0.1 --port $PORT" 2>/dev/null || true; }
trap stop_daemon EXIT

echo "→ work dir $WORK"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "✗ $PY is older than 3.10" >&2; exit 1; }
"$PY" -m venv "$VENV" || exit 1
"$VENV/bin/pip" -q install -U pip || exit 1

echo "→ building wheels from $ROOT"
"$VENV/bin/pip" -q wheel --no-deps -w "$WORK/wheels" "$ROOT/packages/core" "$ROOT/packages/cli" || exit 1
ls -1 "$WORK/wheels"
# core first, so the cli's pfnstudio-core requirement is met by this checkout's
# build rather than the published one.
"$VENV/bin/pip" -q install "$WORK"/wheels/pfnstudio_core-*.whl || exit 1
"$VENV/bin/pip" -q install "$WORK"/wheels/pfnstudio-[0-9]*.whl || exit 1
"$VENV/bin/pip" show pfnstudio pfnstudio-core | grep -E '^(Name|Version):' | paste - -
if "$VENV/bin/python" -c 'import torch' 2>/dev/null; then
  echo "✗ torch is already importable in the fresh venv; the install-torch path would not be tested" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  SLUGS=("$@")
else
  # shellcheck disable=SC2207
  SLUGS=($(curl -fsS "$API/marketplace/listings" | "$VENV/bin/python" -c '
import json, sys
for x in json.load(sys.stdin):
    if isinstance(x, dict) and x.get("type") == "basemodel" and x.get("openSource") is True:
        print(x["slug"])'))
  [[ ${#SLUGS[@]} -gt 0 ]] || { echo "✗ no open base models listed at $API" >&2; exit 1; }
fi
echo "→ models: ${SLUGS[*]}"

# Build input.json — {task, inputs} — from the sample the loaded model
# declares for its first task. Exit 3 when it declares none.
make_input() {
  "$VENV/bin/python" - "$DAEMON/models/$1/capabilities" "$2" <<'PY'
import json, sys, urllib.request

body = json.load(urllib.request.urlopen(sys.argv[1], timeout=120))
r = body.get("result", body) if isinstance(body, dict) else {}
caps = r.get("capabilities") if isinstance(r, dict) else None
if not isinstance(caps, dict) or not caps:
    sys.exit(3)
task = next(iter(caps))
sample = (caps[task] or {}).get("sample")
if not isinstance(sample, dict):
    sys.exit(3)
meta = {"name", "description", "task", "source", "expected"}
inputs = {k: v for k, v in sample.items() if k not in meta}
json.dump({"task": task, "inputs": inputs}, open(sys.argv[2], "w"))
PY
}

# run_case <log> <args...> → prints PASS/FAIL and the seconds it took
run_case() {
  local log="$1"; shift
  local t0=$SECONDS
  ( cd "$WORK" && "$VENV/bin/pfnstudio" run "$@" --install-deps ) >"$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]] && grep -q "Result" "$log"; then
    echo "PASS $((SECONDS - t0))s"
  else
    echo "FAIL $((SECONDS - t0))s"
  fi
}

declare -a ROWS
fail=0
for slug in "${SLUGS[@]}"; do
  echo "── $slug"
  a=$(run_case "$WORK/logs/$slug.sample.log" "$slug")
  echo "   own sample:  $a  (logs/$slug.sample.log)"

  inp="$WORK/inputs/$slug.json"
  b="SKIP no daemon"; c="-"
  if curl -fsS "$DAEMON/healthz" >/dev/null 2>&1; then
    make_input "$slug" "$inp"
    case $? in
      0)
        b=$(run_case "$WORK/logs/$slug.input.log" "$slug" --input "$inp")
        if [[ -n "$SITE" ]]; then
          if curl -fsS "$SITE/models/$slug/input.json" -o "$WORK/inputs/$slug.site.json" 2>/dev/null \
             && "$VENV/bin/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])) == json.load(open(sys.argv[2])) else 1)' \
                  "$inp" "$WORK/inputs/$slug.site.json"; then
            c="MATCH"
          else
            c="FAIL"
          fi
        fi
        ;;
      3) b="SKIP no sample declared" ;;
      *) b="FAIL capabilities unreadable" ;;
    esac
  fi
  echo "   --input:     $b  (logs/$slug.input.log)"
  [[ -n "$SITE" ]] && echo "   site input:  $c"
  ROWS+=("$(printf '%-24s %-14s %-26s %s' "$slug" "$a" "$b" "$c")")
  [[ "$a$b$c" == *FAIL* ]] && fail=1

  # Free the model's memory before the next one loads.
  curl -fsS -X POST "$DAEMON/unload" -H 'Content-Type: application/json' \
    -d "{\"slug\": \"$slug\"}" >/dev/null 2>&1 || true
done

echo
printf '%-24s %-14s %-26s %s\n' MODEL "OWN SAMPLE" "--input input.json" "SITE input.json"
printf '%s\n' "${ROWS[@]}"
echo
echo "logs: $WORK/logs   daemon: $WORK/daemon.log"
if [[ $fail -ne 0 ]]; then
  echo "✗ not ready to publish"
  exit 1
fi
echo "✓ every model ran both ways"
