# Aster & Oak — Campaign Intelligence

Ingests a messy ad-ops campaign export, normalizes and enriches it with an LLM via
**strict tool calling**, stores it in **PostgreSQL**, and serves it back over a
**FastAPI** API.

Built for the Evernine Brands · Hive practical exercise.

---

## Run it

```bash
# 1. Postgres (host port 5433, so it cannot collide with a local install)
docker compose up -d

# 2. Environment
cp .env.example .env
#    then put your key in .env:  ANTHROPIC_API_KEY=sk-ant-...

# 3. Dependencies
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 4. Serve  (tables are created on startup)
./.venv/bin/uvicorn app.main:app --reload
```

Interactive docs: **http://localhost:8000/docs**

```bash
./demo.sh                      # ingest + every endpoint, end to end
./.venv/bin/pytest -q          # 86 tests; needs no API key and no database
```

**It runs without an API key.** With `ANTHROPIC_API_KEY` unset, every campaign is
enriched by the deterministic fallback and tagged `enrichment_source: "fallback"`.
Nothing crashes and nothing is silently mislabelled — which is the same path that
protects the service when the model is available but failing.

### API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/campaigns/ingest` | Ingests `campaigns_raw.json`. Optional body: `{"path": ...}` or `{"payload": {...}}`. Returns the per-row summary. |
| `GET` | `/campaigns` | Filters: `channel`, `objective`, `min_health_score`, `max_health_score`, `min_spend`, `flagged_only`, `needs_review`; plus `sort`, `order`, `limit`, `offset`. |
| `GET` | `/campaigns/{external_id}` | One campaign. |
| `GET` | `/campaigns/insights` | LLM portfolio observations *(stretch)*. |
| `GET` | `/campaigns/runs` · `/campaigns/runs/{id}/rows` | Ingest history and per-row audit trail. |
| `GET` | `/healthz` | Liveness + whether the LLM is configured. |

```bash
curl -X POST localhost:8000/campaigns/ingest
curl "localhost:8000/campaigns?channel=meta&min_health_score=70"
curl "localhost:8000/campaigns?needs_review=true"     # LLM and rules disagreed
curl localhost:8000/campaigns/cmp_012
```

---

## Design choices

### The pipeline

```
raw JSON
   │
   ├─ 1. clean + validate      deterministic, no network   normalize.py
   │       └─ reject junk here, before it costs a token
   ├─ 2. dedupe within batch   first occurrence wins       ingest.py
   ├─ 3. LLM enrichment        strict tool call, 3 attempts     llm.py
   │       └─ on exhaustion → rule-based fallback          enrich.py
   ├─ 4. self-check            alias map vs. model              enrich.py
   └─ 5. upsert on external_id  → idempotent re-ingest     ingest.py
```

**Validation runs before the LLM, deliberately.** The junk row never reaches the
model — it is cheaper, faster, and it means failure handling does not depend on the
model behaving. The model is asked only for judgement (objective, health); anything
a regex can do reliably (`"42,500"` → `42500`) a regex does.

### Schema

Three tables, each with one job:

- **`campaigns`** — the enriched, queryable output. One row per *unique* campaign id,
  with cleaned source fields, derived rates (CTR/CVR/CPA/ROAS, stored so filters stay
  cheap), the LLM's enrichment, the self-check verdict, and a `flags` array.
- **`ingest_runs`** — one row per ingest call.
- **`ingest_row_results`** — one row per *input* row, **including rejected ones**.

That split is the core modelling decision: `campaigns` stays clean enough to trust in
a query, while nothing from the raw export is silently lost. `GET /campaigns/runs/1/rows`
shows exactly what happened to all 15 input rows.

### What I cleaned, rejected and flagged

The brief says there is no single correct cleaning, so here are the rules and why.

**Repair a field, reject a row.** A bad cell kills the cell, not the record.

| Row | Problem | Decision |
|---|---|---|
| `cmp_002` | `spend: "42,500"` | Coerce; flag `spend_was_string`. |
| `cmp_006` | `spend: 0` (Klaviyo) | **Valid.** Owned channel. ROAS is `NULL`, not `0` and not `∞`. |
| `cmp_009` | `impressions: null` | **Valid.** SMS has no impressions. CTR stays `NULL`. |
| `cmp_011` | `name: ""` | **Keep.** Synthesize a name, flag `missing_name`. A cosmetic gap must not discard ₹40k of real spend. |
| `cmp_012` | ₹350k spend, every metric `null` | **Keep + flag `spend_without_attribution`.** The most important row in the file: real money with no attribution. Scored low *because it is unmeasurable*, not because it performed badly — those need opposite responses (fix tracking vs. cut budget). |
| `cmp_013` | `"N/A"`, `"lots"`, `-5`, `"many"` | **Rejected, pre-LLM.** |
| `cmp_008` ×2 | Duplicate id, different numbers | **First wins, second rejected and recorded.** |
| `cmp_014` | `spend: -1200` | **Null the field, keep the row.** Five good fields survive; flag `negative_spend`. |

Three decisions I would defend hardest:

1. **`NULL` and `0` are different facts, and the distinction is load-bearing.**
   SMS has no impressions (`NULL`); an email flow genuinely costs nothing (`0`).
   Every division returns `None` rather than `0` or `inf` when undefined — a zero-spend
   email flow would otherwise rank as either the worst or the best campaign in the
   portfolio, and both are lies. The prompt states this to the model explicitly.

2. **The junk row is rejected on its data, not its name.** The rule is "no usable
   signal: spend, conversions *and* revenue are all unreadable". Matching on
   `"DO NOT USE"` would pass the same test today and break the moment someone renames
   the row — and it would throw away a real campaign that happened to be called
   "Test — new creative". Both cases are covered in the tests.

3. **The duplicate is rejected, not overwritten.** Last-write-wins would quietly
   destroy the evidence that two conflicting records ever existed. "Which of these is
   right?" is a question for ad-ops, so the collision is recorded with the row index
   of the record it collided with.

### Failure handling

A three-rung ladder; a row only descends when the rung above fails:

| Rung | Catches | Behaviour |
|---|---|---|
| 1. Validation | Junk rows | Rejected with a reason. Never reaches the model. |
| 2. LLM, 3 attempts | Transport (429/5xx/timeout) → backoff + jitter.<br>Model answered in prose → re-ask.<br>Arguments out of range → re-ask **with the validation error**.<br>API rejected the tool *schema* → drop `strict`, retry once. | Every other 4xx fails fast — a 401 will still be a 401 on attempt 3. |
| 3. Fallback | Everything else, including no API key | Rule-based enrichment, tagged `enrichment_source: "fallback"` with the error in `enrichment_error`. |

Each campaign is one API call, so a failure is isolated to one row by construction;
calls fan out across a bounded thread pool. **Every campaign gets a channel, an
objective and a score no matter what the model does**, and the row always says which
rung produced it — so a fallback guess is never mistaken for a model judgement.

### Stretch items

- **Idempotent re-ingest** — upsert on `external_id`. Re-running ingest reports
  `created: 0, updated: 13`. Asserted in the demo.
- **Self-check** — `channel` is the one enriched field with deterministic ground
  truth (the alias map), so a mismatch is unambiguously a model error rather than a
  difference of opinion. Where the alias map has no opinion (`"???"`), the verdict is
  `unverified` rather than an invented grade. `health_score` gets a looser band check
  against the rule-based score; only a gap >30 is flagged. Surfaced via
  `GET /campaigns?needs_review=true`.
- **Portfolio insights** — aggregates computed in SQL/Python, the LLM sees *only*
  those aggregates. It never sees raw rows, so it cannot invent a number.
- **Tests** — 86, covering coercion, the rejection rules, guarded division, and every
  LLM failure mode via a scripted fake client, plus a regression test for a
  null-summed-as-zero bug this caught in the insights aggregation. No key or
  database needed.

---

## AI & prompt choices

**Tool calling, not JSON-in-a-prompt.** `strict: true` with
`additionalProperties: false` makes the API reject a malformed call before it reaches
us, and `tool_choice: {type: "tool"}` forces the call. There is no prose to strip, no
` ```json ` fence to regex, no "Sure, here's the JSON!" to survive. Free-text parsing
is exactly the failure mode the brief calls out.

**One Pydantic model is the single source of truth.** `CampaignEnrichment` generates
the tool's JSON Schema *and* re-validates what comes back, so the prompt and the
parser cannot drift apart.

**The strict schema degrades rather than taking the batch down with it.** Per-row
isolation protects against *one* campaign failing, but a rejected tool schema is a
correlated failure — it would fail all 13 rows at the same instant and push the whole
batch onto the fallback path. So a 400 that names the schema drops `strict: true` and
retries once; Pydantic still enforces the contract client-side, so only the
server-side guarantee is lost, and the downgrade is logged. Every other 4xx still
fails fast. Tested.

**Two layers of guarantee, because the wire schema is not enough.** Numeric bounds
(`minimum`/`maximum`/`maxLength`) are not part of the strict-tool subset, so they are
stripped from the wire and enforced by Pydantic on arrival. A model *can* legally
return `health_score: 150`; Pydantic catches it and the retry carries the validation
error back, so attempt 2 is better informed than attempt 1 instead of being a blind
repeat. This is tested.

**Field descriptions do the prompting.** The enum mapping (`fb`/`IG`/`Facebook Ads`
→ `meta`) lives in the `channel` field's description, next to the values it
constrains, rather than in a system prompt paragraph the model has to connect back
to a schema.

**What the system prompt is actually for** — the four judgement rules a schema
cannot express:

1. *Judge each campaign against the norms of its own channel and objective.* Without
   this, every awareness campaign scores badly for having a low ROAS. A YouTube
   awareness buy at 0.69x is doing its job; a Performance Max campaign at 0.69x is on fire.
2. *`NULL` is not zero.* Stated explicitly because the single most likely enrichment
   error on this dataset is reading "not recorded" as "zero" and scoring accordingly.
3. *Unmeasurable is its own failure, and a serious one.* `cmp_012` must not land in
   the middle of the pack. It is scored low **and** told to say *why* — the fix is
   tracking, not budget.
4. *Infer objective from the name and description, never from performance.* A
   conversion campaign that converted badly is still a conversion campaign.

**One call per campaign, not one call for the batch.** It costs more tokens, and it
buys the thing the brief asks for: a malformed response damages exactly one row. A
single batch call puts all 13 campaigns behind one JSON parse — one bad token and the
whole batch is gone.

**Arithmetic is pre-computed, not delegated.** The prompt hands the model CTR, CVR,
CPA and ROAS already calculated, and renders an undefined ratio as
`"undefined (zero media cost — owned channel)"` rather than as a blank. The model is
never left to infer what a missing value means, and it is never asked to do the one
thing it is worse at than a line of Python.

**Model:** `claude-opus-5`, configurable via `LLM_MODEL`. The key is read from the
environment and `.env` is gitignored — nothing secret enters the repo.

---

## What I'd do with more time

In rough priority order:

1. **Verify the live LLM path.** No API key was available while building, so the
   enrichment path is covered by mocked tests of every failure mode but has not run
   against the real API. That is the first thing to do with a key.
2. **Alembic migrations.** `create_all()` on startup is fine for a 90-minute
   exercise and wrong for anything that will ever need a schema change.
3. **Batch the ingest LLM calls through the Message Batches API** — 50% cheaper and
   the right shape for a nightly export rather than an interactive endpoint.
4. **Make ingest async.** It is a synchronous endpoint doing ~13 network calls; it
   should enqueue a job and return a run id to poll.
5. **Persist the raw row alongside the cleaned one** (`raw_payload JSONB`), so
   cleaning decisions can be re-run and audited without the original file.
6. **Prompt-cache the system prompt and tool schema** — identical on every call, so
   it is free throughput currently being paid for 13 times.
7. **Grow the self-check into a proper eval set** — a dozen hand-labelled campaigns
   with expected channel/objective, run in CI so a prompt change that regresses
   classification fails the build.
8. **Pull the alias map out of code into a table** so ad-ops can add a channel
   spelling without a deploy.

---

## Notes on the stack

- Postgres runs on host port **5433**, not 5432, to avoid colliding with a local install.
- `flags` is `JSON`; on Postgres this would be better as `JSONB` with a GIN index once
  the table is large enough for `flagged_only` to need an index rather than a filter.
- Built and tested on Python 3.14.
