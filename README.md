# Medicare Billing Drift Dashboard

A live dashboard for peer-adjusted anomaly detection in Medicare Part B billing, built on
CMS's public **Physician & Other Practitioners — by Provider and Service** dataset
(2017–2024). Nothing is downloaded to disk ahead of time — the app streams and filters
each year's file live, per specialty, when you select it.

## What problem this solves

CMS publishes provider-level billing data every year, but raw dollar totals or service
counts alone don't tell you much: a busy cardiologist and a low-volume one aren't
comparable, and specialty-wide costs drift with inflation and coding changes anyway.
This dashboard asks a narrower, more useful question: **relative to providers who bill
in a similar way, has this provider's billing pattern shifted more than normal
year-over-year?** That's "peer-adjusted drift" — an outlier-detection lens, not an
absolute-cost lens.

## How peer groups are formed

For each specialty and year independently, every provider is described by five
practice-pattern features (not raw dollar totals, so peer groups reflect billing
*style* rather than practice size alone):

- log(total services) and log(total beneficiaries) — case volume
- average Medicare payment per service, inflation-adjusted to 2024 dollars via BLS CPI-U
- the ratio of submitted charge to Medicare-allowed amount
- Shannon entropy of the provider's HCPCS code mix — how concentrated vs. diversified
  their billing is across procedure codes

These are standardized and clustered with k-means (`build_peer_clusters`) into
peer groups sized proportionally to the specialty (roughly one cluster per 25
providers, capped at 8). A separate, finer-grained view is built with
`compute_communities`: a k-nearest-neighbor similarity graph over the same features,
partitioned with Louvain modularity — this is what the "Community Structure" chart
in the Specialty Overview tab shows, and it can split a k-means cluster into sub-groups
that don't fall out of a fixed cluster count.

## How drift is measured — two independent methods

For the most recent two years of data, each provider is scored two separate ways:

1. **Peer-relative z-score.** For each practice-pattern metric, compute the
   provider's year-over-year change, then z-score that change *within their own
   peer cluster* (not against the whole specialty). The per-metric z-scores are
   combined via Stouffer's method into one `combined_z` anomaly score per provider.
2. **Population Stability Index (PSI).** Independent of peer group, this compares
   the provider's *own* distribution of per-service payment amounts between the
   two years. PSI > 0.25 is the standard threshold (borrowed from credit-risk
   monitoring) for a meaningful distributional shift.

### What FDR correction protects against

Running a significance test on every provider in a peer group, one at a time,
guarantees false positives — with enough providers, some will cross a p < 0.05
threshold by chance alone. The z-score p-values are corrected with the
Benjamini-Hochberg procedure (`statsmodels.stats.multitest.multipletests`,
`fdr_bh`) before a provider is flagged (`z_flag`), which controls the *expected
proportion* of false discoveries among all flagged providers, rather than the
per-test error rate.

### What "cross-method agreement" means

`z_flag` (peer-relative) and `psi_flag` (own-distribution shift) are computed
independently, using different data and different statistical machinery. A
provider flagged by only one method could be an artifact of peer-group
composition, a data quirk, or genuine noise. `cross_method_flag` — both methods
agreeing — is the higher-confidence signal surfaced in the KPI strip and used to
color the peer scatter plot.

## SHAP explainability

An XGBoost regressor is trained per specialty (`train_model`) to predict each
provider's `combined_z` anomaly score from their practice-pattern features. This
isn't used to *detect* drift — the z-score/PSI pipeline already did that — it's
used to *explain* it: `build_shap_explainer` + `render_shap_waterfall` show which
specific features (volume, payment per service, coding entropy, etc.) pushed a
given provider's score up or down, so a reviewer can see the "why" behind a flag
in one glance instead of re-deriving it from the raw numbers.

## Analyst memo generation

The "Generate Memo" button sends a provider's drift metrics and top SHAP
contributors to Claude (via the Anthropic API) to draft a short, neutral analyst
memo. It is rate-limited to 10 generations per browser session to control API
spend. **The memo is a drafting aid, not a verified conclusion** — read the
disclaimer below before trusting or forwarding its wording.

## Disclaimer

**Every flag in this dashboard is a statistical lead, not a fraud finding.**
Billing pattern shifts have many legitimate explanations — a provider changing
practice settings, adopting a new procedure, a coding-guideline update, a
temporary staffing change, a shift toward sicker patients, or simple year-to-year
noise in a low-volume specialty. Nothing here has been verified against medical
records, claims audits, or any ground truth, and no NPI shown should be treated
as identified for wrongdoing. This dashboard is a methodology showcase for
peer-adjusted, statistically-corrected anomaly detection on public government
data — it is a starting point for human review, never a substitute for it.

## Architecture notes

- **Live-fetch, no stored files, no bulk downloads.** CMS also publishes this
  dataset as a multi-gigabyte national bulk CSV per year with no server-side
  filter — an early version of this app streamed and filtered that file
  client-side, which took 10+ minutes per year and crashed on Streamlit
  Community Cloud's free tier. `fetch_year` instead calls CMS's real filterable
  JSON API (`data.cms.gov/data-api/v1/dataset/<id>/data`), which supports
  `filter[Rndrng_Prvdr_Type]=<specialty>` server-side, paginated at its hard
  cap of 6,500 rows/request. Results are cached in-process for 24 hours
  (`st.cache_data`), keyed per (year, specialty).
- **Fetching is parallel at two levels, deliberately bounded.** A single
  specialty-year can need dozens of paginated requests (e.g. Podiatry is
  ~173K rows/year, ~27 pages) — sequential pagination, and sequential years
  on top of that, were both directly responsible for cold-start timeouts and
  the "looks frozen" UX even after switching off the bulk CSV. `fetch_year`
  now fetches `PAGE_FETCH_WORKERS` (3) pages at a time within a year, and
  `load_all_years` fetches `FETCH_MAX_WORKERS` (2) years at a time — peak
  concurrent connections is the product of the two (6), sized conservatively
  because higher concurrency previously triggered Streamlit Cloud's CPU
  throttle. Progress is polled every second and reports pages retrieved so
  far, not just "year N of M done", so a slow fetch still looks alive.
- The default fetch window is the most recent `DEFAULT_YEAR_WINDOW` (3)
  years, not the full 8 — fetching all years at once held enough in-flight
  data to exceed the free tier's memory ceiling (an OOM kill with no Python
  traceback, not a code exception). Full history is an opt-in sidebar
  checkbox. The specialty list is ordered smallest-practical-default first
  (`SPECIALTIES[0]` is what a cold instance fetches on first load); very
  large specialties (Internal Medicine, Family Practice) still take longer
  since page/year concurrency is capped for safety, not raw specialty size.
- `build_provider_features`, the PSI step in `compute_drift`, and the
  community-graph edge construction are all vectorized (pandas
  groupby/transform, dict-lookups instead of repeated boolean scans, numpy
  index math) rather than looping in pure Python — those loops were a real
  CPU cost for larger specialties and a plausible contributor to the
  throttle, independent of the fetch itself.
- If first-load latency on Streamlit Community Cloud's free tier is still too
  slow for a good first impression, the next architectural step is a scheduled
  offline job that pre-computes peer clusters, drift, and models and writes the
  results to a small warehouse (e.g., Snowflake) for the app to read — trading
  live-data freshness for instant load.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (already gitignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

Then:

```bash
streamlit run app.py
```
