#!/usr/bin/env bash
#
# Pre-publish check: install THIS checkout's pfnstudio + pfnstudio-core into a
# fresh venv and run every open base model the way the website tells a user
# to — before anything reaches PyPI.
#
#   ./run.sh verify                      every open model in the marketplace
#   ./run.sh verify ceptron-xs-discovery cdfm
#
# Per model, from a cold start (empty venv: no torch, no adapter, no weights):
#   1. pfnstudio run <slug> -y                 the sample the model declares
#   2. pfnstudio run <slug> -y --input sample  the sample input pfnstudio ships
#      (warm: reuses 1's cache)
#   3. every shipped samples/<slug>/<task>.json must equal the sample the
#      running model declares for that task (converted by sample_to_inputs);
#      the declared ones are written to $WORK/declared/ — copy them into
#      packages/cli/pfnstudio/samples/ when a model's sample changes
# A run passes only when the command exits 0 AND prints a result.
#
# Env: PYTHON (default python3, needs 3.10+), PFNSTUDIO_API_URL (marketplace,
# default hosted), VERIFY_BASEMODELS (undeployed adapters, see below),
# VERIFY_DIR (work dir, default a new temp dir;
# kept afterwards for the logs), VERIFY_PORT (daemon port, default 11541 so a
# daemon already on 11440 is never touched).
#
# Needs: RAM for the largest model plus PyTorch; disk for torch + every
# model's packages and weights. Models are unloaded one by one as they finish.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
API="${PFNSTUDIO_API_URL:-https://cloud.pfnstudio.com/api}"
PORT="${VERIFY_PORT:-11541}"
WORK="${VERIFY_DIR:-$(mktemp -d -t pfnstudio-verify.XXXXXX)}"
mkdir -p "$WORK/logs" "$WORK/declared"
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
# VERIFY_BASEMODELS=<dir holding <fmProject>/adapters/>: test adapters before
# they are deployed. A model whose folder is not there still downloads its
# bundle from the marketplace.
if [[ -n "${VERIFY_BASEMODELS:-}" ]]; then
  export PFNSTUDIO_BASEMODELS_DIR="$VERIFY_BASEMODELS"
fi

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

# Write the sample the loaded model declares for each task, as the
# {task, inputs} request run sends, to $WORK/declared/<slug>/<task>.json, and
# compare each with the file pfnstudio ships. Prints MATCH, or what differs.
# Exit 3 when the model declares no sample at all.
check_samples() {
  "$VENV/bin/python" - "$DAEMON/models/$1/capabilities" "$WORK/declared/$1" "$1" <<'PY'
import json, sys, urllib.request
from pathlib import Path

from pfnstudio.samples import _DIR
from pfnstudio_core.serving.samples import sample_to_inputs

body = json.load(urllib.request.urlopen(sys.argv[1], timeout=120))
r = body.get("result", body) if isinstance(body, dict) else {}
caps = r.get("capabilities") if isinstance(r, dict) else None
out, slug = Path(sys.argv[2]), sys.argv[3]
declared = {}
for task, spec in (caps or {}).items():
    sample = (spec or {}).get("sample")
    if isinstance(sample, dict):
        kind = ((spec or {}).get("input_schema") or {}).get("kind")
        declared[task] = {"task": task, "inputs": sample_to_inputs(kind, sample)}
if not declared:
    sys.exit(3)
out.mkdir(parents=True, exist_ok=True)
problems = []
for task, req in declared.items():
    (out / f"{task}.json").write_text(json.dumps(req))
    shipped = _DIR / slug / f"{task}.json"
    if not shipped.is_file():
        problems.append(f"{task} not shipped")
    elif json.loads(shipped.read_text()) != req:
        problems.append(f"{task} differs")
for f in sorted((_DIR / slug).glob("*.json")) if (_DIR / slug).is_dir() else []:
    if f.stem not in declared:
        problems.append(f"{f.stem} shipped, not declared")
print("MATCH" if not problems else "DIFF " + "; ".join(problems))
PY
}

# run_case <log> <args...> → prints PASS/FAIL and the seconds it took
run_case() {
  local log="$1"; shift
  local t0=$SECONDS
  ( cd "$WORK" && "$VENV/bin/pfnstudio" run "$@" -y ) >"$log" 2>&1
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

  b="SKIP no daemon"; c="-"
  if curl -fsS "$DAEMON/healthz" >/dev/null 2>&1; then
    c=$(check_samples "$slug" 2>>"$WORK/logs/$slug.input.log")
    case $? in
      0)
        b=$(run_case "$WORK/logs/$slug.input.log" "$slug" --input sample)
        [[ "$c" == MATCH ]] || c="FAIL $c"
        ;;
      3) b="SKIP no sample declared"; c="-" ;;
      *) b="FAIL capabilities unreadable"; c="-" ;;
    esac
  fi
  echo "   --input sample:     $b  (logs/$slug.input.log)"
  echo "   shipped = declared: $c"
  ROWS+=("$(printf '%-24s %-14s %-26s %s' "$slug" "$a" "$b" "$c")")
  [[ "$a$b$c" == *FAIL* ]] && fail=1

  # Free the model's memory before the next one loads.
  curl -fsS -X POST "$DAEMON/unload" -H 'Content-Type: application/json' \
    -d "{\"slug\": \"$slug\"}" >/dev/null 2>&1 || true
done

echo
printf '%-24s %-14s %-26s %s\n' MODEL "OWN SAMPLE" "--input sample" "SHIPPED = DECLARED"
printf '%s\n' "${ROWS[@]}"
echo
echo "logs: $WORK/logs   daemon: $WORK/daemon.log   declared samples: $WORK/declared"
if [[ $fail -ne 0 ]]; then
  echo "✗ not ready to publish"
  exit 1
fi
echo "✓ every model ran both ways"
