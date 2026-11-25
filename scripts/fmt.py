"""Pretty-printers for demo.sh. Reads JSON on stdin, takes a view name as argv[1]."""
import json
import sys

view = sys.argv[1]
d = json.load(sys.stdin)


def money(v):
    return "        —" if v is None else f"{v:>9,.0f}"


if view == "ingest":
    print(f"run {d['run_id']}  |  {d['total_rows']} rows -> "
          f"{d['ingested']} ingested, {d['rejected']} rejected")
    print(f"enrichment: {d['enriched_by_llm']} by LLM, {d['enriched_by_fallback']} by fallback"
          f"  |  {d['disagreements']} flagged for review  |  {d['duration_seconds']}s\n")
    for r in d["rows"]:
        mark = "OK " if r["status"] == "ingested" else "REJ"
        print(f"  {mark} [{r['row_index']:2}] {str(r['external_id']):8} "
              f"{r['action']:8} {(r['reason'] or '')[:72]}")

elif view == "idempotent":
    print(f"  created={d['created']}  updated={d['updated']}   <- re-running is a no-op")
    if d["created"] != 0:
        sys.exit("  FAIL: re-ingest created duplicate rows")

elif view == "list":
    for c in d:
        print(f"  {c['external_id']}  {c['channel']:10} {c['objective']:13} "
              f"score={c['health_score']:3}  {c['name'][:46]}")

elif view == "worst":
    for c in d:
        print(f"  {c['external_id']}  score={c['health_score']:3}  spend={money(c['spend'])}  "
              f"{(c['health_rationale'] or '')[:78]}")

elif view == "one":
    for k in ("external_id", "name", "channel", "objective", "spend", "revenue", "roas",
              "health_score", "health_rationale", "flags", "enrichment_source",
              "channel_check", "health_check", "health_score_rule_based"):
        print(f"  {k:24} {d[k]}")

elif view == "flags":
    for c in d:
        print(f"  {c['external_id']}  {', '.join(c['flags'])}")

elif view == "review":
    if not d:
        print("  none — the model and the deterministic rules agreed on every campaign")
    for c in d:
        print(f"  {c['external_id']}  llm_score={c['health_score']} "
              f"rule_score={c['health_score_rule_based']}  "
              f"channel={c['channel']} expected={c['channel_expected']}")

elif view == "rejected":
    for r in d:
        if r["status"] == "rejected":
            print(f"  [{r['row_index']:2}] {r['external_id']}: {r['reason']}")

elif view == "insights":
    print(f"  generated_by: {d['generated_by']}  |  {d['campaigns_considered']} campaigns")
    t = d["totals"]
    print(f"  spend={t['total_spend']:,.0f} revenue={t['total_revenue']:,.0f} "
          f"blended_roas={t['blended_roas']} {t['currency']}")
    for o in d["observations"]:
        print(f"   • {o}")
