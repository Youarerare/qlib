import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SampleGenComparison:
    min_history_window: Dict[str, int]
    label_horizon: Dict[str, int]
    label_preprocessing: Dict[str, str]
    dynamic_instruments: Dict[str, str]
    trading_day_filter: Dict[str, str]
    critical_nodes: List[Dict[str, Any]] = field(default_factory=list)


def get_alpha158_flow_mermaid() -> str:
    return """
```mermaid
flowchart TD
    A[Start: Qlib Data Provider] --> B[Load OHLCV for instruments]
    B --> C{Instruments Resolution}
    C -->|Dynamic| C1[Resolve stock pool by date<br/>e.g. CSI300 components change over time]
    C1 --> D[Compute Alpha158 Features<br/>via Qlib Expression Engine]
    D --> D1{Rolling Window Check}
    D1 -->|Window < required| D2[Produce NaN<br/>e.g. first 59 days for 60-day rolling]
    D1 -->|Window >= required| D3[Compute feature value]
    D2 --> E[Compute Label<br/>Ref $close,-2 / Ref $close,-1 - 1]
    D3 --> E
    E --> E1{Label NaN Check}
    E1 -->|Last 2 trading days| E2[Label = NaN]
    E1 -->|Other days| E3[Label = T+1 to T+2 return]
    E2 --> F[Apply Processors]
    E3 --> F
    F --> F1[infer_processors: EMPTY]
    F1 --> F2[learn_processors: DropnaLabel]
    F2 --> F3{Drop rows with NaN label}
    F3 --> F4[learn_processors: CSZScoreNorm label]
    F4 --> G[Final Dataset]

    style D2 fill:#ff9999
    style E2 fill:#ff9999
    style F3 fill:#ffcc00
    style C1 fill:#99ccff
```
"""


def get_alpha50_flow_mermaid() -> str:
    return """
```mermaid
flowchart TD
    A[Start: Pre-computed Pickle File] --> B[Load top50_features.pkl]
    B --> B1{Pickle Coverage Check}
    B1 -->|Stock not in pickle| B2[MISSING - no data<br/>CRITICAL: fixed stock pool]
    B1 -->|Stock in pickle| C[Features already computed<br/>via AlphaCalculatorAuto]
    C --> C1{Rolling Window Check}
    C1 -->|min_periods=1| C2[Produce value even with 1 data point<br/>May be unreliable]
    C1 -->|Formula fails| C3[All NaN column<br/>CRITICAL: entire factor missing]
    C2 --> D[Compute Label<br/>close.pct_change 1 .shift -1]
    C3 --> D
    D --> D1{Label NaN Check}
    D1 -->|Last 1 trading day| D2[Label = NaN]
    D1 -->|Other days| D3[Label = T to T+1 return<br/>DIFFERENT from Alpha158!]
    D2 --> E[Apply Processors]
    D3 --> E
    E --> E1[infer_processors: DropnaLabel]
    E1 --> E2{Drop rows with NaN label<br/>in INFER path - unusual!}
    E2 --> E3[infer_processors: CSZScoreNorm feature]
    E3 --> E4[learn_processors: DropnaLabel<br/>redundant]
    E4 --> E5[learn_processors: CSZScoreNorm label]
    E5 --> F[Final Dataset]

    style B2 fill:#ff6666
    style C2 fill:#ffcc00
    style C3 fill:#ff6666
    style D3 fill:#ff9999
    style E2 fill:#ffcc00
```
"""


def compare_sample_generation() -> SampleGenComparison:
    comparison = SampleGenComparison(
        min_history_window={
            "alpha158": 60,
            "alpha50": 180,
            "description": "Alpha158 needs 60 days max (rolling windows [5,10,20,30,60]). "
                           "Alpha50 needs 180 days max (adv180 calculation). "
                           "Alpha50 with min_periods=1 produces values earlier but they are unreliable.",
        },
        label_horizon={
            "alpha158": 2,
            "alpha50": 1,
            "description": "Alpha158: Ref($close,-2)/Ref($close,-1)-1 uses T+1 to T+2 return (horizon=2). "
                           "Alpha50: pct_change(1).shift(-1) uses T to T+1 return (horizon=1). "
                           "This means Alpha158 loses 2 days at end, Alpha50 loses 1 day. "
                           "The labels are also SHIFTED by 1 day relative to each other!",
        },
        label_preprocessing={
            "alpha158": "CSZScoreNorm (cross-sectional z-score on label, in learn_processors)",
            "alpha50": "CSZScoreNorm (cross-sectional z-score on label, in learn_processors). "
                       "Additionally, CSZScoreNorm is applied to FEATURES in infer_processors.",
            "description": "Both apply CSZScoreNorm to labels. Alpha50 additionally normalizes features "
                           "cross-sectionally in infer_processors.",
        },
        dynamic_instruments={
            "alpha158": "Dynamic - QlibDataLoader resolves instruments at runtime using D.instruments(). "
                        "Stock pool changes over time (e.g., CSI300 rebalancing).",
            "alpha50": "Static - TopkAlphaLoader loads from pre-computed pickle. "
                       "Stock pool is FIXED at computation time. "
                       "If pickle was computed with D.list_instruments() for a specific date range, "
                       "it includes ALL stocks that were in the index at ANY point during that range. "
                       "This is different from Alpha158's dynamic resolution.",
            "description": "This is a MAJOR source of sample count difference. "
                           "Alpha158 dynamically resolves which stocks exist on each date. "
                           "Alpha50 uses a fixed set of stocks from the pickle file.",
        },
        trading_day_filter={
            "alpha158": "No explicit filter. Qlib's data provider naturally excludes non-trading days. "
                        "Stocks with missing data (suspended, delisted) produce NaN which is handled by processors.",
            "alpha50": "No explicit filter. Depends on what was in the pickle file. "
                       "No special handling for limit-up/limit-down, STAR Market, etc.",
            "description": "Neither dataset explicitly filters for special trading conditions. "
                           "However, Alpha158's QlibDataLoader naturally handles suspension via NaN, "
                           "while Alpha50's pre-computed data may or may not include suspended stocks.",
        },
    )

    comparison.critical_nodes = [
        {
            "node_id": "CN1",
            "name": "Label Formula Difference",
            "alpha158": "Ref($close,-2)/Ref($close,-1)-1 (T+1 to T+2 return)",
            "alpha50": "pct_change(1).shift(-1) (T to T+1 return)",
            "impact": "CRITICAL - Different labels mean different NaN patterns (2 vs 1 day lost at end) "
                      "and shifted return periods. This alone causes sample count differences.",
            "fix": "Use identical label formula for both datasets.",
        },
        {
            "node_id": "CN2",
            "name": "Dynamic vs Static Stock Pool",
            "alpha158": "Dynamic instrument resolution per date",
            "alpha50": "Fixed stock pool from pickle",
            "impact": "HIGH - Alpha158 includes/excludes stocks dynamically as index rebalances. "
                      "Alpha50 has a fixed set that may include stocks not in the index on certain dates "
                      "or miss stocks that joined later.",
            "fix": "Re-compute Alpha50 data with dynamic instrument resolution, or filter pickle data "
                   "to match Alpha158's dynamic stock pool.",
        },
        {
            "node_id": "CN3",
            "name": "Feature NaN Policy",
            "alpha158": "Exact window (NaN for insufficient history)",
            "alpha50": "min_periods=1 (values with even 1 data point)",
            "impact": "MEDIUM - Alpha50 produces more non-NaN values in early periods, but they may be "
                      "statistically unreliable. Alpha158's strict NaN policy leads to more sample drops.",
            "fix": "Use consistent NaN policy. Recommend exact window matching Alpha158.",
        },
        {
            "node_id": "CN4",
            "name": "Infer Processor Difference",
            "alpha158": "Empty infer_processors",
            "alpha50": "DropnaLabel + CSZScoreNorm(feature) in infer_processors",
            "impact": "HIGH - Alpha50 drops NaN-label samples in BOTH infer and learn paths. "
                      "Alpha158 only drops in learn path. With PTYPE_A (append), Alpha50's learn data "
                      "goes through infer_processors first, then learn_processors, causing double DropnaLabel.",
            "fix": "Use empty infer_processors for both, and only apply DropnaLabel in learn_processors.",
        },
        {
            "node_id": "CN5",
            "name": "Maximum History Window",
            "alpha158": "60 days",
            "alpha50": "180 days (for adv180)",
            "impact": "LOW - With min_periods=1, Alpha50 doesn't lose samples from this. "
                      "But with exact window policy, Alpha50 would lose 179 days at start vs 59 for Alpha158.",
            "fix": "If using exact window policy, start data earlier to accommodate the larger window.",
        },
    ]

    return comparison


def print_sample_generation_analysis():
    comparison = compare_sample_generation()

    print("\n" + "=" * 80)
    print("SAMPLE GENERATION LOGIC ANALYSIS")
    print("=" * 80)

    print("\n--- Alpha158 Sample Generation Flow ---")
    print(get_alpha158_flow_mermaid())

    print("\n--- Alpha50 Sample Generation Flow ---")
    print(get_alpha50_flow_mermaid())

    print("\n--- Detailed Comparison ---")

    aspects = [
        ("Minimum History Window", comparison.min_history_window),
        ("Label Horizon", comparison.label_horizon),
        ("Label Preprocessing", comparison.label_preprocessing),
        ("Dynamic Instruments", comparison.dynamic_instruments),
        ("Trading Day Filter", comparison.trading_day_filter),
    ]

    for name, data in aspects:
        print(f"\n  [{name}]")
        print(f"    Alpha158: {data['alpha158']}")
        print(f"    Alpha50:  {data['alpha50']}")
        print(f"    Analysis: {data['description']}")

    print("\n--- Critical Nodes (Sample Count Difference Sources) ---")
    for node in comparison.critical_nodes:
        print(f"\n  [{node['node_id']}] {node['name']}")
        print(f"    Alpha158: {node['alpha158']}")
        print(f"    Alpha50:  {node['alpha50']}")
        print(f"    Impact:   {node['impact']}")
        print(f"    Fix:      {node['fix']}")

    print("\n" + "=" * 80)
