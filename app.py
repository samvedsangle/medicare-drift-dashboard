import concurrent.futures

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
import xgboost as xgb
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

st.set_page_config(
    page_title="Medicare Billing Drift Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Year -> CMS data-api dataset id (data.cms.gov's real filterable JSON API
# for this dataset, NOT the bulk CSV distribution). The bulk CSVs for this
# dataset are multi-gigabyte national files with no server-side filter,
# which is far too slow for a live per-specialty fetch; this API supports
# filter[Rndrng_Prvdr_Type]=<specialty> server-side, at a hard cap of
# API_PAGE_SIZE rows per request (paginated via `offset` in fetch_year).
DATASETS = {
    2024: "92396110-2aed-4d63-a6a2-5d6207d46a29",
    2023: "0e9f2f2b-7bf9-451a-912c-e02e654dd725",
    2022: "e650987d-01b7-4f09-b75e-b0b075afbf98",
    2021: "31dc2c47-f297-4948-bfb4-075e1bec3a02",
    2020: "c957b49e-1323-49e7-8678-c09da387551d",
    2019: "867b8ac7-ccb7-4cc9-873d-b24340d89e32",
    2018: "fb6d9fe8-38c1-4d24-83d4-0b7b291000b2",
    2017: "85bf3c9c-2244-490d-ad7d-c34e4c28f8ea",
}
API_PAGE_SIZE = 6500

# Canonical name -> raw CMS column name. Verified live against the 2016, 2017
# and 2024 distributions: CMS has back-normalized every reissued year to this
# same header set, so a single mapping covers the whole DATASETS range.
SCHEMA_MAP = {
    "npi": "Rndrng_NPI",
    "last_name": "Rndrng_Prvdr_Last_Org_Name",
    "first_name": "Rndrng_Prvdr_First_Name",
    "city": "Rndrng_Prvdr_City",
    "state": "Rndrng_Prvdr_State_Abrvtn",
    "specialty": "Rndrng_Prvdr_Type",
    "hcpcs_code": "HCPCS_Cd",
    "hcpcs_desc": "HCPCS_Desc",
    "place_of_service": "Place_Of_Srvc",
    "tot_benes": "Tot_Benes",
    "tot_srvcs": "Tot_Srvcs",
    "tot_bene_day_srvcs": "Tot_Bene_Day_Srvcs",
    "avg_submitted_chrg": "Avg_Sbmtd_Chrg",
    "avg_medicare_allowed": "Avg_Mdcr_Alowd_Amt",
    "avg_medicare_payment": "Avg_Mdcr_Pymt_Amt",
    "avg_medicare_stdzd": "Avg_Mdcr_Stdzd_Amt",
}

REQUIRED_COLS = list(SCHEMA_MAP.keys())

NUMERIC_COLS = [
    "tot_benes",
    "tot_srvcs",
    "tot_bene_day_srvcs",
    "avg_submitted_chrg",
    "avg_medicare_allowed",
    "avg_medicare_payment",
    "avg_medicare_stdzd",
]

DOLLAR_COLS = [
    "avg_submitted_chrg",
    "avg_medicare_allowed",
    "avg_medicare_payment",
    "avg_medicare_stdzd",
]

# BLS CPI-U annual averages (all items, US city average, 1982-84=100).
# Used to convert every year's dollar figures into constant BASE_YEAR
# dollars, so "drift" reflects behavioral change, not inflation.
CPI = {
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}
BASE_YEAR = 2024

# Ordered smallest-practical-default first: the selectbox default (index 0)
# is what a cold Streamlit Cloud instance fetches on first load, and the
# biggest specialties (Internal Medicine, Family Practice) are more likely
# to hit the free tier's memory/time ceiling on a cold start.
SPECIALTIES = [
    "Podiatry",
    "Optometry",
    "Chiropractic",
    "Physical Therapist",
    "Dermatology",
    "Ophthalmology",
    "Psychiatry",
    "Rheumatology",
    "Pain Management",
    "Nephrology",
    "Pulmonary Disease",
    "Gastroenterology",
    "Urology",
    "Neurology",
    "Anesthesiology",
    "General Surgery",
    "Orthopedic Surgery",
    "Diagnostic Radiology",
    "Cardiology",
    "Emergency Medicine",
    "Nurse Practitioner",
    "Physician Assistant",
    "Family Practice",
    "Internal Medicine",
]

DRIFT_METRICS = ("payment_per_service", "hcpcs_entropy", "submitted_allowed_ratio", "log_tot_srvcs")
METRIC_LABELS = {
    "payment_per_service": "Payment per service",
    "hcpcs_entropy": "Billing-code diversity (entropy)",
    "submitted_allowed_ratio": "Charge / allowed ratio",
    "log_tot_srvcs": "Volume (log total services)",
}
MODEL_FEATURES = [
    "log_tot_srvcs",
    "log_tot_benes",
    "payment_per_service",
    "submitted_allowed_ratio",
    "hcpcs_entropy",
    "n_hcpcs_codes",
]
PSI_FLAG_THRESHOLD = 0.25
FDR_ALPHA = 0.05
MEMO_RATE_LIMIT = 10
# Default cold-start fetch is a smaller trailing window of years, not the
# full configured history: fetching all years concurrently, each with its
# own paginated sequence of requests, held enough in-flight data at once to
# exceed Streamlit Community Cloud's free-tier memory ceiling (an OOM kill,
# not a code exception — the process died with no traceback). Full history
# is available as an opt-in from the sidebar.
DEFAULT_YEAR_WINDOW = 3
FETCH_MAX_WORKERS = 2
PAGE_FETCH_WORKERS = 3


# ---------------------------------------------------------------------------
# Data fetch (cached, live — nothing written to disk)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24, max_entries=12)
def fetch_year(year: int, specialty: str, _page_counter: dict | None = None) -> pd.DataFrame:
    """Fetch one year's data for one specialty via CMS's data-api, which
    filters server-side (filter[Rndrng_Prvdr_Type]=<specialty>) — the bulk
    CSV distribution for this dataset is a multi-gigabyte national file with
    no server-side filter, which made a live per-specialty fetch far too
    slow (minutes per year, regardless of parsing engine).

    Pages are fetched PAGE_FETCH_WORKERS at a time rather than one at a
    time: a specialty needing ~27 pages was paying ~27 sequential
    round-trips for no reason, since CMS's API has no problem serving
    several requests concurrently. Combined with FETCH_MAX_WORKERS years
    running concurrently in `load_all_years`, peak concurrent connections
    is PAGE_FETCH_WORKERS x FETCH_MAX_WORKERS — kept modest on purpose so
    this doesn't re-trigger Streamlit Cloud's CPU throttle. Each batch is
    processed in offset order and stops at the first short/empty page, so
    pagination correctness doesn't depend on request completion order.
    Nothing is written to disk.

    `_page_counter` (leading underscore so st.cache_data excludes it from
    the cache key) is an optional shared dict this bumps after every page,
    so a caller running this in a background thread can report live
    within-year progress rather than only "year N of M done"."""
    dataset_id = DATASETS[year]
    base_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
    raw_to_canon = {v: k for k, v in SCHEMA_MAP.items()}

    def fetch_page(offset: int):
        params = {
            "filter[Rndrng_Prvdr_Type]": specialty,
            "size": API_PAGE_SIZE,
            "offset": offset,
        }
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        page = resp.json()
        if _page_counter is not None:
            _page_counter[year] = _page_counter.get(year, 0) + 1
        return offset, page

    records = []
    next_offset = 0
    done = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=PAGE_FETCH_WORKERS) as pool:
        while not done:
            batch_offsets = [next_offset + i * API_PAGE_SIZE for i in range(PAGE_FETCH_WORKERS)]
            # Submit all of the batch first (eager list), *then* collect
            # results — submitting inside the generator sorted() consumes
            # would call .result() on each future before the next is even
            # submitted, silently serializing everything.
            futures = [pool.submit(fetch_page, o) for o in batch_offsets]
            batch = sorted((f.result() for f in futures), key=lambda x: x[0])
            for _offset, page in batch:
                if not page:
                    done = True
                    break
                records.extend(page)
                if len(page) < API_PAGE_SIZE:
                    done = True
                    break
            next_offset += PAGE_FETCH_WORKERS * API_PAGE_SIZE

    if not records:
        return pd.DataFrame(columns=REQUIRED_COLS + ["year"] + [f"{c}_adj" for c in DOLLAR_COLS])

    df = pd.DataFrame.from_records(records)
    df = df.rename(columns=raw_to_canon)[REQUIRED_COLS]

    df["npi"] = df["npi"].astype(str)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year"] = year
    inflation_factor = CPI[BASE_YEAR] / CPI[year]
    for c in DOLLAR_COLS:
        df[f"{c}_adj"] = df[c] * inflation_factor

    return df


def load_all_years(specialty: str, years: tuple) -> pd.DataFrame:
    """Not itself cached — `fetch_year` is cached per (year, specialty).
    Each year requires its own paginated sequence of API calls (a busy
    specialty can need dozens of pages), so years are fetched concurrently
    rather than one after another: sequential 8-year fetches were the
    direct cause of the 600s+ cold-start timeouts seen on Streamlit Cloud.

    Progress is polled (not just updated on each year's completion) and
    reports pages retrieved so far across all in-flight years — otherwise
    the bar sits frozen at "Starting fetch…" for the entire duration of
    whichever year finishes first, which under Streamlit Cloud's CPU
    throttling can be minutes and looks indistinguishable from a hang."""
    frames = []
    progress = st.progress(0.0, text="Starting fetch…")
    page_counts: dict = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_to_year = {executor.submit(fetch_year, yr, specialty, page_counts): yr for yr in years}
        pending = set(future_to_year)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=1.0)
            for future in done:
                yr = future_to_year[future]
                try:
                    frames.append(future.result())
                except (OSError, ValueError, requests.exceptions.RequestException) as exc:
                    st.warning(f"Could not load {yr} ({exc}); continuing with remaining years.")
                completed += 1
            total_pages = sum(page_counts.values())
            progress.progress(
                completed / len(years),
                text=f"Fetched {completed}/{len(years)} years — {total_pages} pages retrieved so far…",
            )
    progress.empty()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_provider_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse provider-service-line rows into one row per (npi, year).
    Fully vectorized (groupby/transform, no per-group Python loop) — a
    per-group Python loop here was a real CPU hot spot for larger
    specialties, contributing to Streamlit Cloud's CPU throttling.
    `state` uses each provider-year's first row rather than a true mode,
    trading a rare multi-state-in-one-year edge case for a big speedup."""
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.loc[raw_df["tot_srvcs"] > 0].copy()
    df["_w_payment"] = df["avg_medicare_payment_adj"] * df["tot_srvcs"]
    df["_w_allowed"] = df["avg_medicare_allowed_adj"] * df["tot_srvcs"]
    df["_w_submitted"] = df["avg_submitted_chrg_adj"] * df["tot_srvcs"]

    agg = df.groupby(["npi", "year"], as_index=False).agg(
        specialty=("specialty", "first"),
        state=("state", "first"),
        last_name=("last_name", "first"),
        first_name=("first_name", "first"),
        city=("city", "first"),
        tot_srvcs=("tot_srvcs", "sum"),
        tot_benes=("tot_benes", "sum"),
        w_payment=("_w_payment", "sum"),
        w_allowed=("_w_allowed", "sum"),
        w_submitted=("_w_submitted", "sum"),
        n_hcpcs_codes=("hcpcs_code", "nunique"),
    )

    agg["log_tot_srvcs"] = np.log1p(agg["tot_srvcs"])
    agg["log_tot_benes"] = np.log1p(agg["tot_benes"])
    agg["payment_per_service"] = agg["w_payment"] / agg["tot_srvcs"]
    allowed_per_service = agg["w_allowed"] / agg["tot_srvcs"]
    submitted_per_service = agg["w_submitted"] / agg["tot_srvcs"]
    agg["submitted_allowed_ratio"] = np.where(
        allowed_per_service > 0, submitted_per_service / allowed_per_service, np.nan
    )

    code_sums = df.groupby(["npi", "year", "hcpcs_code"])["tot_srvcs"].sum().reset_index(name="code_srvcs")
    code_totals = code_sums.groupby(["npi", "year"])["code_srvcs"].transform("sum")
    p = code_sums["code_srvcs"] / code_totals
    code_sums["p_logp"] = np.where(p > 0, -p * np.log(p), 0.0)
    entropy = (
        code_sums.groupby(["npi", "year"], as_index=False)["p_logp"]
        .sum()
        .rename(columns={"p_logp": "hcpcs_entropy"})
    )

    out = agg.merge(entropy, on=["npi", "year"], how="left")
    return out[
        [
            "npi", "year", "specialty", "state", "last_name", "first_name", "city",
            "tot_srvcs", "tot_benes", "log_tot_srvcs", "log_tot_benes",
            "payment_per_service", "submitted_allowed_ratio", "hcpcs_entropy", "n_hcpcs_codes",
        ]
    ]


# ---------------------------------------------------------------------------
# Peer clustering
# ---------------------------------------------------------------------------

CLUSTER_FEATURES = ["log_tot_srvcs", "log_tot_benes", "payment_per_service", "hcpcs_entropy", "submitted_allowed_ratio"]


def build_peer_clusters(features: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Assign each provider a peer_cluster within their own year, so a
    provider is only ever compared against others billing at a similar
    volume and case mix in that same year."""
    if features.empty:
        return features

    parts = []
    for year, grp in features.groupby("year"):
        g = grp.copy()
        n = len(g)
        if n < 50:
            g["peer_cluster"] = 0
        else:
            X = g[CLUSTER_FEATURES].fillna(g[CLUSTER_FEATURES].median())
            this_k = max(2, min(k, n // 25))
            X_scaled = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=this_k, n_init=10, random_state=42)
            g["peer_cluster"] = km.fit_predict(X_scaled)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Drift detection: peer-relative z-scores (with FDR correction) plus PSI
# ---------------------------------------------------------------------------

def psi(expected, actual, buckets: int = 10) -> float:
    """Population Stability Index between two distributions. <0.1 = stable,
    0.1-0.25 = moderate shift, >0.25 = significant shift (standard credit-risk
    convention, reused here for billing-pattern shift)."""
    # Plain numpy (not pd.Series) — this runs once per provider, and building
    # two Series per call dominated the drift step's runtime.
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return np.nan

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_drift(raw_df: pd.DataFrame, features: pd.DataFrame, metrics=DRIFT_METRICS) -> pd.DataFrame:
    """Two independent flags per provider, for the latest year-over-year
    transition available:
      1. z_flag  — peer-relative: is this provider's YoY change in billing
         pattern an outlier *within their own peer cluster*, after
         Benjamini-Hochberg FDR correction across the whole peer group.
      2. psi_flag — individual: has this provider's own service-level
         payment distribution shifted meaningfully between the two years,
         independent of any peer comparison.
    cross_method_flag (both agree) is the higher-confidence signal."""
    years = sorted(features["year"].unique())
    if len(years) < 2:
        return pd.DataFrame()

    y0, y1 = years[-2], years[-1]
    prev = features[features.year == y0].set_index("npi")
    curr = features[features.year == y1].set_index("npi")
    common = prev.index.intersection(curr.index)
    if len(common) == 0:
        return pd.DataFrame()

    result = pd.DataFrame(index=common)
    result["peer_cluster"] = curr.loc[common, "peer_cluster"]

    z_cols = []
    for m in metrics:
        delta = curr.loc[common, m] - prev.loc[common, m]
        result[f"{m}_delta"] = delta
        z = pd.Series(index=common, dtype=float)
        for _, idx in result.groupby("peer_cluster").groups.items():
            grp_delta = delta.loc[idx]
            mu = grp_delta.mean()
            sigma = grp_delta.std(ddof=0)
            sigma = sigma if sigma and sigma > 1e-9 else 1e-9
            z.loc[idx] = (grp_delta - mu) / sigma
        result[f"{m}_z"] = z
        z_cols.append(f"{m}_z")

    # Stouffer's method: combine per-metric z-scores into one anomaly score.
    result["combined_z"] = result[z_cols].mean(axis=1) * np.sqrt(len(z_cols))
    result["combined_p"] = 2 * stats.norm.sf(np.abs(result["combined_z"]))
    _, p_adj, _, _ = multipletests(result["combined_p"].fillna(1.0), alpha=FDR_ALPHA, method="fdr_bh")
    result["fdr_p"] = p_adj
    result["z_flag"] = result["fdr_p"] < FDR_ALPHA

    # Pre-filter to common NPIs and pre-group both years once — looking up
    # curr_raw with a fresh boolean mask inside the loop (curr_raw.npi ==
    # npi_val) was an O(n_providers x n_rows) scan, a real CPU hot spot for
    # larger specialties. Dict-of-groups lookups make each iteration O(1).
    prev_common_raw = raw_df.loc[(raw_df.year == y0) & raw_df["npi"].isin(common)]
    curr_common_raw = raw_df.loc[(raw_df.year == y1) & raw_df["npi"].isin(common)]
    curr_groups = dict(tuple(curr_common_raw.groupby("npi")))

    psi_scores = {}
    for npi_val, grp0 in prev_common_raw.groupby("npi"):
        grp1 = curr_groups.get(npi_val)
        if grp1 is not None and len(grp0) >= 5 and len(grp1) >= 5:
            psi_scores[npi_val] = psi(grp0["avg_medicare_payment_adj"], grp1["avg_medicare_payment_adj"])
    result["psi_score"] = pd.Series(psi_scores)
    result["psi_flag"] = result["psi_score"] > PSI_FLAG_THRESHOLD

    result["cross_method_flag"] = result["z_flag"].fillna(False) & result["psi_flag"].fillna(False)
    result["npi"] = result.index
    result["baseline_year"] = y0
    result["current_year"] = y1
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Community detection (finer-grained sub-structure within peer clusters)
# ---------------------------------------------------------------------------

def compute_communities(features: pd.DataFrame, year: int, k_neighbors: int = 8):
    sub = features[features.year == year].dropna(subset=CLUSTER_FEATURES)
    if len(sub) < 10:
        return nx.Graph(), {}

    X = StandardScaler().fit_transform(sub[CLUSTER_FEATURES])
    n_neighbors = min(k_neighbors + 1, len(sub))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    dist, idx = nn.kneighbors(X)

    npis = sub["npi"].values
    # Vectorize the neighbor-index/weight bookkeeping with numpy instead of
    # a nested Python double loop — add_edges_from still costs one Python
    # call per edge (networkx graphs are Python objects), but this removes
    # the per-element index math that dominated for larger specialties.
    src_idx = np.repeat(np.arange(len(sub)), n_neighbors - 1)
    dst_idx = idx[:, 1:].ravel()
    weights = 1.0 / (1.0 + dist[:, 1:].ravel())

    G = nx.Graph()
    G.add_nodes_from(npis)
    G.add_weighted_edges_from(zip(npis[src_idx], npis[dst_idx], weights))

    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
    community_map = {}
    for cid, members in enumerate(communities):
        for m in members:
            community_map[m] = cid
    return G, community_map


# ---------------------------------------------------------------------------
# Model + SHAP explainability
# ---------------------------------------------------------------------------

def train_model(curr_features: pd.DataFrame, drift: pd.DataFrame, feature_cols=MODEL_FEATURES):
    if drift.empty:
        return None, None, None
    merged = curr_features.merge(drift[["npi", "combined_z"]], on="npi", how="inner").dropna(subset=["combined_z"])
    if len(merged) < 30:
        return None, None, None

    X = merged[feature_cols].fillna(merged[feature_cols].median())
    y = merged["combined_z"]
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    return model, X.reset_index(drop=True), merged["npi"].values


def build_shap_explainer(model, X: pd.DataFrame):
    if model is None:
        return None
    return shap.TreeExplainer(model)


def render_shap_waterfall(explainer, X: pd.DataFrame, npi_array, target_npi: str):
    if explainer is None or target_npi not in set(npi_array):
        st.info("Not enough peer-group data to compute a SHAP explanation for this provider.")
        return None
    idx = int(np.where(npi_array == target_npi)[0][0])
    row = X.iloc[[idx]]
    sv = explainer(row)
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.plots.waterfall(sv[0], show=False, max_display=8)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    return sv[0]


# ---------------------------------------------------------------------------
# Chart / KPI renderers
# ---------------------------------------------------------------------------

ANALYSIS_CACHE_SIZE = 2


@st.cache_resource(show_spinner=False)
def get_analysis_store() -> dict:
    """Process-wide store of finished analyses, keyed by (specialty, years).
    Streamlit reruns the whole script on every click, and the analysis
    (clustering, drift/PSI, community detection, model training) took ~30s —
    so switching providers cost ~30s each time even though only the selected
    NPI changed. Kept as a small bounded dict (see ANALYSIS_CACHE_SIZE) so
    memory stays inside the free tier."""
    return {}


def analyze(raw_df: pd.DataFrame) -> dict:
    features = build_peer_clusters(build_provider_features(raw_df))
    drift = compute_drift(raw_df, features, DRIFT_METRICS)
    latest_year = int(features["year"].max())
    _, community_map = compute_communities(features, latest_year)
    curr_features = features[features.year == latest_year].copy()
    model, model_X, model_npis = train_model(curr_features, drift, MODEL_FEATURES)
    return {
        "features": features,
        "drift": drift,
        "latest_year": latest_year,
        "community_map": community_map,
        "curr_features": curr_features,
        "model": model,
        "model_X": model_X,
        "model_npis": model_npis,
        "explainer": build_shap_explainer(model, model_X),
    }


def _index_to_first(values) -> np.ndarray:
    """Rescale a series so its first value is 100 — puts metrics with very
    different units (dollars vs. an entropy score) on one comparable axis so
    drift is actually visible instead of one metric dwarfing the rest."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr[0]) or arr[0] == 0:
        return np.full(arr.shape, np.nan)
    return arr / arr[0] * 100


def _style_year_axis(fig: go.Figure) -> None:
    # dtick=1 + integer format: otherwise plotly invents "2,022.5"-style ticks.
    fig.update_xaxes(title_text="Year", dtick=1, tickformat="d")


def render_peer_scatter(features: pd.DataFrame, drift: pd.DataFrame, year: int, selected_npi: str):
    sub = features[features.year == year]
    if drift is not None and not drift.empty:
        sub = sub.merge(drift[["npi", "combined_z", "cross_method_flag"]], on="npi", how="left")
    else:
        sub = sub.assign(combined_z=np.nan, cross_method_flag=False)
    if sub.empty:
        st.info("No data available for this year.")
        return

    scored = sub[sub["combined_z"].notna()]
    unscored = sub[sub["combined_z"].isna()]

    fig = go.Figure()
    if not unscored.empty:
        fig.add_trace(
            go.Scatter(
                x=unscored["log_tot_srvcs"],
                y=unscored["payment_per_service"],
                mode="markers",
                marker=dict(color="lightgray", size=5, opacity=0.5),
                customdata=unscored[["npi"]].to_numpy(),
                hovertemplate="NPI %{customdata[0]}<br>No prior-year data<extra></extra>",
                name="No prior-year data",
            )
        )
    if not scored.empty:
        fig.add_trace(
            go.Scatter(
                x=scored["log_tot_srvcs"],
                y=scored["payment_per_service"],
                mode="markers",
                marker=dict(
                    color=scored["combined_z"],
                    colorscale="RdBu_r",
                    cmin=-4,
                    cmax=4,
                    size=6,
                    opacity=0.75,
                    colorbar=dict(title="Drift z-score"),
                ),
                customdata=scored[["npi", "peer_cluster", "combined_z"]].to_numpy(),
                hovertemplate=(
                    "NPI %{customdata[0]}<br>Peer cluster %{customdata[1]}"
                    "<br>Drift z-score %{customdata[2]:.2f}<extra></extra>"
                ),
                name="Providers",
                showlegend=False,
            )
        )
    row = sub[sub.npi == selected_npi]
    if not row.empty:
        fig.add_trace(
            go.Scatter(
                x=row["log_tot_srvcs"],
                y=row["payment_per_service"],
                mode="markers",
                marker=dict(size=18, color="black", symbol="star"),
                name="Selected provider",
            )
        )
    fig.update_layout(
        title=f"Peer landscape — {year}",
        xaxis_title="Log(total services)",
        yaxis_title="Avg payment / service ($, inflation-adjusted)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")


def render_provider_trajectory(features: pd.DataFrame, selected_npi: str):
    traj = features[features.npi == selected_npi].sort_values("year")
    fig = go.Figure()
    for m in DRIFT_METRICS:
        fig.add_trace(
            go.Scatter(
                x=traj["year"],
                y=_index_to_first(traj[m]),
                customdata=traj[m].to_numpy(),
                mode="lines+markers",
                name=METRIC_LABELS[m],
                hovertemplate="%{x}: index %{y:.1f} (raw %{customdata:.3g})<extra>" + METRIC_LABELS[m] + "</extra>",
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Provider trajectory (indexed: first year = 100)",
        yaxis_title="Index (first year = 100)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    _style_year_axis(fig)
    st.plotly_chart(fig, width="stretch")


def render_kpi_strip(features: pd.DataFrame, drift: pd.DataFrame, year: int):
    n_providers = features[features.year == year]["npi"].nunique()
    n_flagged = int(drift["cross_method_flag"].sum()) if drift is not None and not drift.empty else 0
    pct_flagged = (n_flagged / len(drift) * 100) if drift is not None and len(drift) else 0.0
    median_z = float(drift["combined_z"].median()) if drift is not None and not drift.empty else 0.0

    cols = st.columns(4)
    cols[0].metric("Providers analyzed", f"{n_providers:,}")
    cols[1].metric("Cross-method flags", f"{n_flagged:,}")
    cols[2].metric("% flagged", f"{pct_flagged:.2f}%")
    cols[3].metric("Median drift z-score", f"{median_z:.2f}")


def render_drift_trend(features: pd.DataFrame, metrics=DRIFT_METRICS):
    trend = features.groupby("year")[list(metrics)].median().reset_index().sort_values("year")
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(
            go.Scatter(
                x=trend["year"],
                y=_index_to_first(trend[m]),
                customdata=trend[m].to_numpy(),
                mode="lines+markers",
                name=METRIC_LABELS.get(m, m),
                hovertemplate="%{x}: index %{y:.1f} (raw %{customdata:.3g})<extra>"
                + METRIC_LABELS.get(m, m)
                + "</extra>",
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Specialty-wide median billing pattern (indexed: first year = 100)",
        yaxis_title="Index (first year = 100)",
        legend_title="Metric",
    )
    _style_year_axis(fig)
    st.plotly_chart(fig, width="stretch")


def generate_memo(provider_row: pd.Series, drift_row, shap_explanation) -> str:
    import anthropic

    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured in .streamlit/secrets.toml")

    client = anthropic.Anthropic(api_key=api_key)

    drift_summary = "no significant peer-relative or distributional drift detected"
    if drift_row is not None:
        drift_summary = (
            f"combined peer-relative z-score {drift_row['combined_z']:.2f} "
            f"(FDR-adjusted p={drift_row['fdr_p']:.4f}, flagged={bool(drift_row['z_flag'])}); "
            f"PSI vs. prior year {drift_row['psi_score']:.3f} (flagged={bool(drift_row['psi_flag'])}); "
            f"cross-method agreement={bool(drift_row['cross_method_flag'])}"
        )

    shap_summary = "not available"
    if shap_explanation is not None:
        order = np.argsort(-np.abs(shap_explanation.values))[:3]
        parts = [
            f"{shap_explanation.feature_names[i]} ({shap_explanation.values[i]:+.2f})" for i in order
        ]
        shap_summary = "; ".join(parts)

    prompt = f"""You are drafting an internal analyst memo about a single Medicare provider's billing pattern.
This is a statistical lead for further review, NOT a fraud finding or accusation.

Provider NPI: {provider_row.get('npi')}
Specialty: {provider_row.get('specialty')}
State: {provider_row.get('state')}
Total services (latest year): {provider_row.get('tot_srvcs'):,.0f}
Avg payment per service (real $): {provider_row.get('payment_per_service'):.2f}
Distinct HCPCS codes billed: {provider_row.get('n_hcpcs_codes')}

Drift signal: {drift_summary}
Top SHAP contributors to the anomaly score: {shap_summary}

Write a concise (150-200 word) neutral analyst memo summarizing what changed in this
provider's billing pattern relative to their peers, what might statistically explain it,
and what a reviewer should look at next. Do not assert fraud or wrongdoing; frame findings
as statistical leads that warrant human review."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🩺 Medicare Billing Drift Dashboard")
st.caption(
    "Peer-adjusted anomaly detection over live CMS Physician & Other Practitioners data. "
    "Flags here are statistical leads, not fraud findings."
)

with st.sidebar:
    st.header("Filters")
    specialty = st.selectbox("Specialty", SPECIALTIES, index=0)
    all_years = tuple(sorted(DATASETS.keys()))
    recent_years = all_years[-DEFAULT_YEAR_WINDOW:]
    full_history = st.checkbox(
        f"Load full {len(all_years)}-year history (slower, more memory)",
        value=False,
        help=f"Default is the most recent {DEFAULT_YEAR_WINDOW} years for a fast first load.",
    )
    years = all_years if full_history else recent_years
    st.caption(f"Years analyzed: {years[0]}–{years[-1]}")

analysis_key = (specialty, years)
analysis_store = get_analysis_store()
if analysis_key not in analysis_store:
    with st.spinner(f"Fetching live CMS data for {specialty} ({years[0]}-{years[-1]})… first load takes a while."):
        raw_df = load_all_years(specialty, years)
    if raw_df.empty:
        st.error("No rows returned for this specialty. Check the specialty name against CMS's Rndrng_Prvdr_Type values.")
        st.stop()
    with st.spinner("Analyzing peer groups, drift and the explanation model…"):
        analysis_store[analysis_key] = analyze(raw_df)
    while len(analysis_store) > ANALYSIS_CACHE_SIZE:
        analysis_store.pop(next(iter(analysis_store)))

_analysis = analysis_store[analysis_key]
features = _analysis["features"]
drift = _analysis["drift"]
latest_year = _analysis["latest_year"]
community_map = _analysis["community_map"]
model, model_X, model_npis, explainer = (
    _analysis["model"],
    _analysis["model_X"],
    _analysis["model_npis"],
    _analysis["explainer"],
)
# Copy: the analysis dict is shared across reruns and sessions, and the
# Specialty Overview tab adds a column to this frame.
curr_features = _analysis["curr_features"].copy()

with st.sidebar:
    top_providers = curr_features.sort_values("tot_srvcs", ascending=False).head(500)
    options = top_providers["npi"].tolist()
    label_map = {
        row.npi: f"{row.npi} — {row.last_name}, {row.first_name} ({row.city}, {row.state})"
        for row in top_providers.itertuples()
    }
    manual_npi = st.text_input("Or enter an exact NPI", value="")
    if manual_npi and manual_npi in curr_features["npi"].values:
        selected_npi = manual_npi
    else:
        selected_npi = st.selectbox(
            "Provider (top 500 by volume)",
            options,
            format_func=lambda n: label_map.get(n, n),
        )

render_kpi_strip(features, drift, latest_year)

tab_provider, tab_specialty = st.tabs(["Provider Detail", "Specialty Overview"])

with tab_provider:
    provider_row = curr_features[curr_features.npi == selected_npi]
    drift_row = None
    if not drift.empty:
        match = drift[drift.npi == selected_npi]
        if not match.empty:
            drift_row = match.iloc[0]

    st.subheader(f"Provider {selected_npi}")
    if not provider_row.empty:
        r = provider_row.iloc[0]
        st.write(f"**{r.last_name}, {r.first_name}** — {r.city}, {r.state} — {r.specialty}")

    col_traj, col_scatter = st.columns(2)
    with col_traj:
        render_provider_trajectory(features, selected_npi)

    with col_scatter:
        render_peer_scatter(features, drift, latest_year, selected_npi)

    st.subheader("SHAP: What Drives This Provider's Drift Score")
    shap_explanation = None
    if model is not None:
        shap_explanation = render_shap_waterfall(explainer, model_X, model_npis, selected_npi)
    else:
        st.info("Not enough providers in this specialty/year to train a drift model.")

    st.subheader("Generate Analyst Memo")
    clicks = st.session_state.get("memo_clicks", 0)
    st.caption(f"{clicks}/{MEMO_RATE_LIMIT} memos generated this session.")
    if st.button("Generate Memo"):
        if clicks >= MEMO_RATE_LIMIT:
            st.error("Rate limit reached for this session (10 memos). Refresh the page to reset.")
        elif provider_row.empty:
            st.error("No feature data available for this provider.")
        else:
            st.session_state["memo_clicks"] = clicks + 1
            try:
                with st.spinner("Generating memo…"):
                    memo_text = generate_memo(provider_row.iloc[0], drift_row, shap_explanation)
                st.session_state["last_memo"] = memo_text
                st.session_state["last_memo_npi"] = selected_npi
            except Exception as exc:
                st.error(f"Memo generation failed: {exc}")
    # Only show a memo if it was generated for the CURRENTLY selected
    # provider — otherwise switching providers without re-clicking left the
    # previous provider's memo text displayed under the new provider's
    # section with nothing indicating it was stale/mismatched.
    if st.session_state.get("last_memo_npi") == selected_npi and "last_memo" in st.session_state:
        st.markdown(st.session_state["last_memo"])

with tab_specialty:
    st.subheader("Drift Trend")
    render_drift_trend(features)

    st.subheader("Peer Cluster Distribution (latest year)")
    cluster_counts = curr_features["peer_cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts, x_label="Peer cluster", y_label="Providers")

    st.subheader("Community Structure (latest year)")
    if community_map:
        curr_features["community"] = curr_features["npi"].map(community_map)
        community_counts = curr_features["community"].value_counts().sort_index()
        st.bar_chart(community_counts, x_label="Community", y_label="Providers")
        st.caption(
            f"{community_counts.shape[0]} communities detected among {len(community_map):,} providers "
            "via Louvain modularity on a k-nearest-neighbor similarity graph of practice-pattern features."
        )
    else:
        st.info("Not enough providers to build a community graph for this specialty/year.")
