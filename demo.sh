#!/usr/bin/env bash
# Live walkthrough demo. Assumes the API is running on :8000 (override with BASE).
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"
PY="${PY:-python3}"
F="$PY scripts/fmt.py"
h() { printf '\n\033[1;36m── %s\033[0m\n' "$*"; }

h "Health"
curl -s "$BASE/healthz" | $PY -m json.tool

h "Ingest — 15 raw rows, per-row outcome"
curl -s -X POST "$BASE/campaigns/ingest" | $F ingest

h "Idempotency — re-ingest must create nothing"
curl -s -X POST "$BASE/campaigns/ingest" | $F idempotent

h "GET /campaigns?channel=meta&min_health_score=70"
curl -s "$BASE/campaigns?channel=meta&min_health_score=70" | $F list

h "GET /campaigns?min_spend=100000&order=asc — worst big spenders"
curl -s "$BASE/campaigns?min_spend=100000&sort=health_score&order=asc&limit=3" | $F worst

h "GET /campaigns/cmp_012 — ₹350k spent, zero attribution"
curl -s "$BASE/campaigns/cmp_012" | $F one

h "GET /campaigns?flagged_only=true — the data-quality trail"
curl -s "$BASE/campaigns?flagged_only=true&limit=50" | $F flags

h "GET /campaigns?needs_review=true — LLM vs rule-based disagreement"
curl -s "$BASE/campaigns?needs_review=true" | $F review

h "Rejected rows are recorded, not lost — GET /campaigns/runs/1/rows"
curl -s "$BASE/campaigns/runs/1/rows" | $F rejected

h "Error handling"
printf "  GET /campaigns/does_not_exist   -> "; curl -s -o /dev/null -w "%{http_code}\n" "$BASE/campaigns/does_not_exist"
printf "  GET /campaigns?channel=tiktok   -> "; curl -s -o /dev/null -w "%{http_code}\n" "$BASE/campaigns?channel=tiktok"
printf "  GET /campaigns?min_health_score=500 -> "; curl -s -o /dev/null -w "%{http_code}\n" "$BASE/campaigns?min_health_score=500"

h "Portfolio insights (stretch)"
curl -s "$BASE/campaigns/insights" | $F insights

printf '\n\033[1;32mDemo complete.\033[0m\n'
