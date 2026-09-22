import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

DATASETS = {
    2024: "https://data.cms.gov/sites/default/files/2026-05/b5ebab5a-f490-418a-9bce-4b9f31419356/PHY_R26_P05_V10_D24_Prov_Svc.csv",
    2023: "https://data.cms.gov/sites/default/files/2025-04/e3f823f8-db5b-4cc7-ba04-e7ae92b99757/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv",
    2022: "https://data.cms.gov/sites/default/files/2025-11/53fb2bae-4913-48dc-a6d4-d8c025906567/MUP_PHY_R25_P05_V20_D22_Prov_Svc.csv",
    2021: "https://data.cms.gov/sites/default/files/2025-11/bffaf97a-c2ab-4fd7-8718-be90742e3485/MUP_PHY_R25_P05_V20_D21_Prov_Svc.csv",
    2020: "https://data.cms.gov/sites/default/files/2025-11/d22b18cd-7726-4bf5-8e9c-3e4587c589a1/MUP_PHY_R25_P05_V20_D20_Prov_Svc.csv",
    2019: "https://data.cms.gov/sites/default/files/2025-11/7befba27-752e-47a8-a76c-6c6d4f74f2e3/MUP_PHY_R25_P04_V20_D19_Prov_Svc.csv",
    2018: "https://data.cms.gov/sites/default/files/2025-11/5669eafb-f0b3-4dc5-be6d-abc09b480c2e/MUP_PHY_R25_P04_V20_D18_Prov_Svc.csv",
    2017: "https://data.cms.gov/sites/default/files/2025-11/4623fb40-781e-4eef-860e-b851cd5d10ea/MUP_PHY_R25_P04_V20_D17_Prov_Svc.csv",
}

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


# ---------------------------------------------------------------------------
# Data fetch (cached, live — nothing written to disk)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_year(year: int, specialty: str) -> pd.DataFrame:
    """Stream one year's CMS provider-service CSV and keep only rows for
    `specialty`, so peak memory stays bounded regardless of the full file
    size (these run into the hundreds of MB per year)."""
    url = DATASETS[year]
    raw_to_canon = {v: k for k, v in SCHEMA_MAP.items()}
    usecols = list(SCHEMA_MAP.values())

    chunks = []
    # Let pandas manage the HTTP stream directly (via urllib) rather than
    # wrapping requests' raw socket in a TextIOWrapper — on multi-hundred-MB
    # chunked-transfer files the manual wrapper's connection got torn down
    # mid-read ("I/O operation on closed file"); pandas' own remote-CSV path
    # doesn't have that failure mode.
    reader = pd.read_csv(url, usecols=usecols, chunksize=250_000, low_memory=False)
    for chunk in reader:
        chunk = chunk.rename(columns=raw_to_canon)
        filtered = chunk.loc[chunk["specialty"] == specialty]
        if not filtered.empty:
            chunks.append(filtered.copy())

    if not chunks:
        return pd.DataFrame(columns=REQUIRED_COLS + ["year"] + [f"{c}_adj" for c in DOLLAR_COLS])

    df = pd.concat(chunks, ignore_index=True)
    df["npi"] = df["npi"].astype(str)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year"] = year
    inflation_factor = CPI[BASE_YEAR] / CPI[year]
    for c in DOLLAR_COLS:
        df[f"{c}_adj"] = df[c] * inflation_factor

    return df


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_all_years(specialty: str, years: tuple) -> pd.DataFrame:
    frames = []
    for yr in years:
        try:
            frames.append(fetch_year(yr, specialty))
        except (OSError, ValueError) as exc:
            st.warning(f"Could not load {yr} ({exc}); continuing with remaining years.")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_provider_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse provider-service-line rows into one row per (npi, year)."""
    if raw_df.empty:
        return pd.DataFrame()

    def _entropy(counts: np.ndarray) -> float:
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    records = []
    for (npi_val, year), grp in raw_df.groupby(["npi", "year"]):
        tot_srvcs = grp["tot_srvcs"].sum()
        tot_benes = grp["tot_benes"].sum()
        if tot_srvcs <= 0:
            continue
        payment_per_service = (grp["avg_medicare_payment_adj"] * grp["tot_srvcs"]).sum() / tot_srvcs
        allowed_per_service = (grp["avg_medicare_allowed_adj"] * grp["tot_srvcs"]).sum() / tot_srvcs
        submitted_per_service = (grp["avg_submitted_chrg_adj"] * grp["tot_srvcs"]).sum() / tot_srvcs
        submitted_allowed_ratio = submitted_per_service / allowed_per_service if allowed_per_service else np.nan
        code_volumes = grp.groupby("hcpcs_code")["tot_srvcs"].sum().values
        records.append(
            {
                "npi": npi_val,
                "year": year,
                "specialty": grp["specialty"].iloc[0],
                "state": grp["state"].mode().iat[0] if not grp["state"].mode().empty else np.nan,
                "last_name": grp["last_name"].iloc[0],
                "first_name": grp["first_name"].iloc[0],
                "city": grp["city"].iloc[0],
                "tot_srvcs": tot_srvcs,
                "tot_benes": tot_benes,
                "log_tot_srvcs": np.log1p(tot_srvcs),
                "log_tot_benes": np.log1p(tot_benes),
                "payment_per_service": payment_per_service,
                "submitted_allowed_ratio": submitted_allowed_ratio,
                "hcpcs_entropy": _entropy(code_volumes),
                "n_hcpcs_codes": grp["hcpcs_code"].nunique(),
            }
        )
    return pd.DataFrame(records)


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
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
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

    psi_scores = {}
    prev_raw = raw_df[raw_df.year == y0]
    curr_raw = raw_df[raw_df.year == y1]
    for npi_val, grp0 in prev_raw.groupby("npi"):
        if npi_val not in common:
            continue
        grp1 = curr_raw[curr_raw.npi == npi_val]
        if len(grp0) >= 5 and len(grp1) >= 5:
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
    G = nx.Graph()
    G.add_nodes_from(npis)
    for i in range(len(sub)):
        for j_pos in range(1, idx.shape[1]):
            j = idx[i, j_pos]
            weight = 1.0 / (1.0 + dist[i, j_pos])
            G.add_edge(npis[i], npis[j], weight=weight)

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

def render_peer_scatter(features: pd.DataFrame, drift: pd.DataFrame, year: int, selected_npi: str):
    sub = features[features.year == year]
    if drift is not None and not drift.empty:
        sub = sub.merge(drift[["npi", "combined_z", "cross_method_flag"]], on="npi", how="left")
    if sub.empty:
        st.info("No data available for this year.")
        return

    fig = px.scatter(
        sub,
        x="log_tot_srvcs",
        y="payment_per_service",
        color=sub["combined_z"] if "combined_z" in sub else None,
        color_continuous_scale="RdBu_r",
        hover_data=["npi", "peer_cluster"],
        labels={"log_tot_srvcs": "Log(Total Services)", "payment_per_service": "Avg Payment / Service ($, real)"},
        title=f"Peer Landscape — {year}",
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
    trend = features.groupby("year")[list(metrics)].median().reset_index()
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Scatter(x=trend["year"], y=trend[m], mode="lines+markers", name=m))
    fig.update_layout(
        title="Specialty-Wide Median Billing Pattern Over Time",
        xaxis_title="Year",
        yaxis_title="Median value (metric-specific units)",
        legend_title="Metric",
    )
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
    years = tuple(sorted(DATASETS.keys()))
    st.caption(f"Years analyzed: {years[0]}–{years[-1]}")

years_key = years

with st.spinner(f"Fetching live CMS data for {specialty} ({years[0]}-{years[-1]})… first load takes a while."):
    raw_df = load_all_years(specialty, years_key)

if raw_df.empty:
    st.error("No rows returned for this specialty. Check the specialty name against CMS's Rndrng_Prvdr_Type values.")
    st.stop()

features = build_provider_features(raw_df)
features = build_peer_clusters(features)
drift = compute_drift(raw_df, features, DRIFT_METRICS)
latest_year = int(features["year"].max())
_, community_map = compute_communities(features, latest_year)

curr_features = features[features.year == latest_year].copy()
model, model_X, model_npis = train_model(curr_features, drift, MODEL_FEATURES)
explainer = build_shap_explainer(model, model_X)

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
        traj = features[features.npi == selected_npi].sort_values("year")
        fig = go.Figure()
        for m in DRIFT_METRICS:
            fig.add_trace(go.Scatter(x=traj["year"], y=traj[m], mode="lines+markers", name=m))
        fig.update_layout(title="Provider Trajectory Across Years", xaxis_title="Year")
        st.plotly_chart(fig, width="stretch")

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
            except Exception as exc:
                st.error(f"Memo generation failed: {exc}")
    if "last_memo" in st.session_state:
        st.markdown(st.session_state["last_memo"])

with tab_specialty:
    st.subheader("Drift Trend")
    render_drift_trend(features)

    st.subheader("Peer Cluster Distribution (latest year)")
    cluster_counts = curr_features["peer_cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Community Structure (latest year)")
    if community_map:
        curr_features["community"] = curr_features["npi"].map(community_map)
        community_counts = curr_features["community"].value_counts().sort_index()
        st.bar_chart(community_counts)
        st.caption(
            f"{community_counts.shape[0]} communities detected among {len(community_map):,} providers "
            "via Louvain modularity on a k-nearest-neighbor similarity graph of practice-pattern features."
        )
    else:
        st.info("Not enough providers to build a community graph for this specialty/year.")
