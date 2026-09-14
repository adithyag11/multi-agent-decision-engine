"""
Deterministic scoring engine: every number in a final decision traces back
to a function in this file, never to an LLM's arithmetic.

Design decision worth calling out explicitly: we do NOT give agents a
general-purpose Python REPL / `exec()` tool. A sandboxed REPL is the
textbook answer to "let the agent compute things," but for a regulated
financial workflow it's the wrong trade: (a) truly isolating arbitrary code
execution needs a microVM/gVisor-class sandbox, which is infrastructure this
service shouldn't own just to add two numbers, and (b) even with perfect
isolation, an auditor cannot pre-certify code an LLM writes fresh on every
run -- there's nothing to point to and say "this is the approved formula."

Instead, agents call named, versioned, unit-testable functions from a fixed
registry (FORMULA_REGISTRY at the bottom). The LLM's job is limited to: (1)
deciding *which* registered formula applies, and (2) supplying its numeric
inputs, validated by Pydantic before the formula ever runs. That's "rigid
tool-calling" in the sense the brief asks for -- the computation itself is
ordinary, reviewable, testable Python, identical to what a quant risk team
would sign off on.
"""
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field


# =============================================================================
# Fraud risk: Beneish M-Score
#
# Beneish, M. D. (1999), "The Detection of Earnings Manipulation," Financial
# Analysts Journal. Eight accounting ratios, each comparing the current
# fiscal year to the prior one, combined into a single index. The published
# threshold: M > -1.78 flags a meaningfully elevated probability of earnings
# manipulation. This is a well-known *screening* heuristic, not a proof of
# fraud -- the Risk/Critic agent treats it as one input among several, and
# the decision output always says so.
#
# One simplification from the original paper: LVGI classically separates
# current liabilities from long-term debt; our source schema only carries
# total_liabilities, so we index total_liabilities/total_assets year over
# year instead. This is a common applied simplification and is labeled as
# such in the result's `notes` field, not silently substituted.
# =============================================================================

class FinancialPeriod(BaseModel):
    fiscal_year: int
    revenue: float = Field(gt=0)
    cogs: float = Field(ge=0)
    net_income: float
    total_assets: float = Field(gt=0)
    current_assets: float = Field(ge=0)
    current_liabilities: float = Field(ge=0)   # for Altman working-capital ratio
    ppe_net: float = Field(ge=0)
    total_liabilities: float = Field(ge=0)
    retained_earnings: float                    # for Altman X2; can be negative (accumulated deficit)
    ebit: float                                  # earnings before interest & tax, for Altman X3; can be negative
    receivables: float = Field(ge=0)
    depreciation: float = Field(ge=0)
    sga_expense: float = Field(ge=0)
    operating_cash_flow: float


class FraudRiskResult(BaseModel):
    m_score: float
    risk_level: str  # 'low' | 'medium' | 'high' | 'critical'
    component_indices: dict[str, float]
    flagged: bool
    notes: list[str]


def compute_beneish_m_score(current: FinancialPeriod, prior: FinancialPeriod) -> FraudRiskResult:
    notes: list[str] = []

    dsri = (current.receivables / current.revenue) / (prior.receivables / prior.revenue)

    prior_gross_margin = (prior.revenue - prior.cogs) / prior.revenue
    current_gross_margin = (current.revenue - current.cogs) / current.revenue
    gmi = prior_gross_margin / current_gross_margin if current_gross_margin != 0 else float("inf")

    prior_aqi_base = 1 - (prior.current_assets + prior.ppe_net) / prior.total_assets
    current_aqi_base = 1 - (current.current_assets + current.ppe_net) / current.total_assets
    aqi = current_aqi_base / prior_aqi_base if prior_aqi_base != 0 else float("inf")

    sgi = current.revenue / prior.revenue

    prior_depi_rate = prior.depreciation / (prior.ppe_net + prior.depreciation) if (prior.ppe_net + prior.depreciation) else 0
    current_depi_rate = current.depreciation / (current.ppe_net + current.depreciation) if (current.ppe_net + current.depreciation) else 0
    depi = prior_depi_rate / current_depi_rate if current_depi_rate != 0 else float("inf")

    sgai = (current.sga_expense / current.revenue) / (prior.sga_expense / prior.revenue)

    tata = (current.net_income - current.operating_cash_flow) / current.total_assets

    notes.append("LVGI uses total_liabilities/total_assets (simplified from the original "
                 "current-liabilities + long-term-debt split; source data does not separate them).")
    prior_leverage = prior.total_liabilities / prior.total_assets
    current_leverage = current.total_liabilities / current.total_assets
    lvgi = current_leverage / prior_leverage if prior_leverage != 0 else float("inf")

    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )

    flagged = m_score > -1.78
    if flagged:
        risk_level = "critical" if m_score > -1.0 else "high"
    else:
        risk_level = "medium" if m_score > -2.5 else "low"

    return FraudRiskResult(
        m_score=round(m_score, 4),
        risk_level=risk_level,
        component_indices={
            "DSRI": round(dsri, 4), "GMI": round(gmi, 4), "AQI": round(aqi, 4),
            "SGI": round(sgi, 4), "DEPI": round(depi, 4), "SGAI": round(sgai, 4),
            "TATA": round(tata, 4), "LVGI": round(lvgi, 4),
        },
        flagged=flagged,
        notes=notes,
    )


# =============================================================================
# Financial distress: Altman Z''-Score (private-firm variant)
#
# Altman, E. I., Hartzell, J., & Peck, M. (1995), "Emerging Markets
# Corporate Bonds: A Scoring System," Salomon Brothers -- the private-firm /
# non-manufacturer adaptation of Altman's original 1968 Z-Score. It
# substitutes book value of equity for market value of equity (the original
# X4), since a private counterparty has no traded share price, and drops
# the asset-turnover term the public-company version includes.
#
# This is deliberately a SECOND, INDEPENDENT signal alongside Beneish, not a
# replacement for it: Beneish asks "does this year's accounting look
# manipulated relative to last year's," a year-over-year comparison. Altman
# asks "is this company financially distressed right now," a point-in-time
# solvency measure needing only the current period. A company can score
# clean on one and be flagged by the other -- e.g. a business in real
# financial distress with no evidence of manipulating its books, or a
# healthy balance sheet with year-over-year red flags -- so the Risk/Critic
# agent is given both rather than one score standing in for "fraud risk"
# on its own. Relying on a single 1999 academic heuristic for a decision
# with real financial consequences is exactly the kind of single-point-of-
# failure a real risk function would reject.
#
# Published zones: Z'' > 2.6 = "safe," 1.1 < Z'' <= 2.6 = "grey," Z'' <= 1.1
# = "distress."
# =============================================================================

class DistressRiskResult(BaseModel):
    z_score: float
    zone: str  # 'safe' | 'grey' | 'distress'
    component_ratios: dict[str, float]
    notes: list[str]


def compute_altman_zscore(period: FinancialPeriod) -> DistressRiskResult:
    working_capital = period.current_assets - period.current_liabilities
    x1 = working_capital / period.total_assets
    x2 = period.retained_earnings / period.total_assets
    x3 = period.ebit / period.total_assets

    book_equity = period.total_assets - period.total_liabilities
    notes = ["Uses book value of equity (total_assets - total_liabilities), not market value -- "
             "the private-firm variant, since a private counterparty has no traded share price."]
    x4 = book_equity / period.total_liabilities if period.total_liabilities != 0 else float("inf")

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if z > 2.6:
        zone = "safe"
    elif z > 1.1:
        zone = "grey"
    else:
        zone = "distress"

    return DistressRiskResult(
        z_score=round(z, 4),
        zone=zone,
        component_ratios={
            "X1_working_capital_to_assets": round(x1, 4),
            "X2_retained_earnings_to_assets": round(x2, 4),
            "X3_ebit_to_assets": round(x3, 4),
            "X4_equity_to_liabilities": round(x4, 4),
        },
        notes=notes,
    )


# =============================================================================
# ESG composite score
#
# IMPORTANT PROVENANCE NOTE, unlike Beneish and Altman above: there is no
# single published, citable "the ESG formula" the way there is for
# earnings-manipulation or distress screening. Commercial raters (MSCI,
# Sustainalytics, and similar) use proprietary methodologies they do not
# publish. What follows is a transparent, reasonable rubric built in that
# general style (bucket each disclosed KPI against thresholds, weight three
# pillars, map to a letter band) -- it is illustrative, not an industry-
# standard methodology anyone could cite externally. Before this is used
# for a real decision, `ESG_RUBRIC_DEFAULT` below is exactly what a real
# ESG policy team needs to replace with their institution's own weights,
# thresholds, and (ideally) licensed third-party scores -- which is why it
# is one importable, swappable config object rather than constants inlined
# through the function body.
# =============================================================================

class EsgDisclosure(BaseModel):
    fiscal_year: int
    revenue: float = Field(gt=0)
    scope1_emissions_tco2e: float = Field(ge=0)
    scope2_emissions_tco2e: float = Field(ge=0)
    board_independence_pct: float = Field(ge=0, le=100)
    workforce_injury_rate: float = Field(ge=0)  # incidents per 200,000 hours worked (OSHA-style)
    controversy_flag_count: int = Field(ge=0)


class EsgWeights(BaseModel):
    environmental: float = 0.4
    social: float = 0.3
    governance: float = 0.3


class EsgRubricConfig(BaseModel):
    """Everything about the rubric that's a policy choice rather than a
    computation, bundled into one object so it can be swapped at the call
    site (or eventually loaded from a config file / admin UI) without
    touching compute_esg_composite's logic at all."""
    weights: EsgWeights = EsgWeights()
    # Each threshold list is [(boundary, score), ...] ascending on boundary;
    # see _band_score's docstring for how direction is encoded in the data.
    emissions_intensity_thresholds: list[tuple[float, float]] = [(50, 100), (150, 70), (400, 40), (1000, 10)]
    injury_rate_thresholds: list[tuple[float, float]] = [(0.5, 100), (2.0, 70), (5.0, 40), (10.0, 10)]
    board_independence_thresholds: list[tuple[float, float]] = [(30, 10), (50, 40), (70, 70), (90, 100)]
    controversy_penalty_per_flag: float = 15
    controversy_penalty_cap: float = 60
    rating_bands: list[tuple[float, str]] = [(80, "AAA"), (65, "AA"), (50, "A"), (35, "BB"), (20, "B"), (0, "CCC")]


ESG_RUBRIC_DEFAULT = EsgRubricConfig()


class EsgScoreResult(BaseModel):
    composite_score: float  # 0-100
    rating: str  # 'AAA'..'CCC'
    pillar_scores: dict[str, float]
    notes: list[str]


def _band_score(value: float, thresholds: list[tuple[float, float]]) -> float:
    """Piecewise-linear score in [0, 100] against ordered (boundary, score)
    checkpoints. `thresholds` must be sorted ascending on the boundary
    value; the score at each checkpoint already encodes the metric's
    direction -- e.g. governance thresholds rise (low independence -> low
    score), emissions thresholds fall (low emissions -> high score). Do NOT
    reverse or otherwise transform the list here: the boundary clamps below
    (`value <= thresholds[0][0]` -> `thresholds[0][1]`, and symmetrically at
    the top) are correct regardless of which direction the scores run, since
    the direction is already baked into the (boundary, score) pairs
    themselves. An earlier version took a `higher_is_better` flag and
    reversed the list for "lower is better" metrics -- that double-applied
    the direction (once via the authored score order, once via the reversal)
    and silently inverted every environmental/social score. Keep this
    function direction-agnostic; put direction only in the threshold data.
    """
    if value <= thresholds[0][0]:
        return thresholds[0][1]
    if value >= thresholds[-1][0]:
        return thresholds[-1][1]
    for (b0, s0), (b1, s1) in zip(thresholds, thresholds[1:]):
        if b0 <= value <= b1:
            span = (b1 - b0) or 1.0
            return s0 + (s1 - s0) * ((value - b0) / span)
    return thresholds[-1][1]


def compute_esg_composite(disclosure: EsgDisclosure, rubric: EsgRubricConfig | None = None) -> EsgScoreResult:
    rubric = rubric or ESG_RUBRIC_DEFAULT
    weights = rubric.weights
    notes: list[str] = []

    emissions_intensity = (disclosure.scope1_emissions_tco2e + disclosure.scope2_emissions_tco2e) / (disclosure.revenue / 1_000_000)
    environmental_score = _band_score(emissions_intensity, rubric.emissions_intensity_thresholds)

    injury_score = _band_score(disclosure.workforce_injury_rate, rubric.injury_rate_thresholds)
    controversy_penalty = min(disclosure.controversy_flag_count * rubric.controversy_penalty_per_flag, rubric.controversy_penalty_cap)
    social_score = max(injury_score - controversy_penalty, 0)
    if controversy_penalty > 0:
        notes.append(f"Social score reduced by {controversy_penalty} points for "
                      f"{disclosure.controversy_flag_count} recorded controversy flag(s).")

    governance_score = _band_score(disclosure.board_independence_pct, rubric.board_independence_thresholds)

    composite = (
        weights.environmental * environmental_score
        + weights.social * social_score
        + weights.governance * governance_score
    )

    # rating_bands is ordered highest-boundary-first; take the first band
    # the composite clears (falls through to the last entry, "CCC" at 0, if
    # somehow below every other boundary).
    rating = next(label for boundary, label in rubric.rating_bands if composite >= boundary)

    return EsgScoreResult(
        composite_score=round(composite, 2),
        rating=rating,
        pillar_scores={
            "environmental": round(environmental_score, 2),
            "social": round(social_score, 2),
            "governance": round(governance_score, 2),
        },
        notes=notes,
    )


# =============================================================================
# Registry: the ONLY surface agents call through. Each entry pairs a stable
# string key (what the LLM's tool-call selects) with its Pydantic input
# schema (what gets validated before the function runs) and the pure
# function itself. Adding a new formula means adding one entry here plus a
# unit test -- never editing agent prompts to "explain the new math."
# =============================================================================

class FormulaSpec(BaseModel):
    key: str
    description: str


class BeneishInputs(BaseModel):
    """Wraps the two-period comparison Beneish needs into a single input
    schema, since the registry's calling convention is `fn(validated_input)`."""
    current: FinancialPeriod
    prior: FinancialPeriod


def _run_beneish(inputs: BeneishInputs) -> FraudRiskResult:
    return compute_beneish_m_score(inputs.current, inputs.prior)


FORMULA_REGISTRY: dict[str, tuple[Callable, type[BaseModel]]] = {
    "beneish_m_score": (_run_beneish, BeneishInputs),
    "altman_zscore_private": (compute_altman_zscore, FinancialPeriod),
    "esg_composite_weighted": (compute_esg_composite, EsgDisclosure),
}
