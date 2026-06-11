"""Machine profiles and per-machine decision engines.

Bridges the simulator's MACHINE_CONFIGS and the financial parameters seeded
in the settings table into the MachineProfile / DecisionEngine inputs, so
scenario costs reflect the actual factory instead of placeholder numbers.
"""
import logging

from src.data_generator.machines import MACHINE_CONFIGS
from src.decision.decision_engine import DecisionEngine, MachineProfile

logger = logging.getLogger(__name__)

# Must mirror scripts/init_db.py seed_default_settings().
FINANCIAL_DEFAULTS: dict[str, float] = {
    "line_a_hourly_production_eur": 850.0,
    "line_b_hourly_production_eur": 720.0,
    "ac_emergency_repair_eur": 2850.0,
    "hx_emergency_repair_eur": 1800.0,
    "cm_emergency_repair_eur": 1400.0,
    "planned_vs_emergency_ratio": 0.35,
    "emergency_premium_mult": 1.8,
}

# Production flow per line: AC feeds HX feeds CM. A stopped machine idles
# everything downstream of it on the same line.
DOWNSTREAM: dict[str, list[str]] = {
    "AC-201": ["HX-202", "CM-203"],
    "HX-202": ["CM-203"],
    "CM-203": [],
    "AC-301": ["HX-302", "CM-303"],
    "HX-302": ["CM-303"],
    "CM-303": [],
}

LABOR_SHARE = 1.0 / 3.0  # planned repair budget split labor vs parts


def load_financials(db=None) -> dict[str, float]:
    """Financial parameters from the settings table, defaults as fallback."""
    financials = dict(FINANCIAL_DEFAULTS)
    if db is None:
        return financials
    try:
        from src.database.models import Settings

        rows = db.query(Settings).filter(Settings.category == "financial").all()
        for row in rows:
            try:
                financials[row.key] = float(row.value)
            except (TypeError, ValueError):
                continue
    except Exception as exc:
        logger.warning(f"Financial settings unavailable, using defaults: {exc}")
    return financials


def production_rate(machine_id: str, financials: dict[str, float]) -> float:
    config = MACHINE_CONFIGS.get(machine_id, {})
    key = (
        "line_a_hourly_production_eur"
        if config.get("line", "A") == "A"
        else "line_b_hourly_production_eur"
    )
    return financials.get(key, FINANCIAL_DEFAULTS[key])


def emergency_repair_cost(machine_id: str, financials: dict[str, float]) -> float:
    machine_type = MACHINE_CONFIGS.get(machine_id, {}).get("type", "AC").lower()
    key = f"{machine_type}_emergency_repair_eur"
    return financials.get(key, FINANCIAL_DEFAULTS.get(key, 2850.0))


def build_profile(
    machine_id: str,
    financials: dict[str, float],
    bearing_stage: str = "II",
) -> MachineProfile:
    """MachineProfile from simulator config + financial settings."""
    config = MACHINE_CONFIGS.get(machine_id, {})
    weibull = config.get("weibull", {})
    return MachineProfile(
        machine_id=machine_id,
        machine_type=config.get("type", "AC"),
        production_rate_per_hour=production_rate(machine_id, financials),
        cascade_targets=DOWNSTREAM.get(machine_id, []),
        weibull_beta=weibull.get("beta", 2.0),
        weibull_eta=weibull.get("eta", 144.0),
        bearing_stage=bearing_stage,
    )


def build_engine(machine_id: str, financials: dict[str, float]) -> DecisionEngine:
    """DecisionEngine whose repair costs match this machine's type.

    The settings table defines the emergency repair cost per machine type
    and the planned/emergency ratio, so:
      planned repair = emergency × ratio,  multiplier = 1 / ratio.
    """
    emergency = emergency_repair_cost(machine_id, financials)
    ratio = financials.get(
        "planned_vs_emergency_ratio", FINANCIAL_DEFAULTS["planned_vs_emergency_ratio"]
    )
    ratio = min(max(ratio, 0.05), 1.0)
    planned_total = emergency * ratio
    return DecisionEngine(
        planned_labor_cost=planned_total * LABOR_SHARE,
        planned_parts_cost=planned_total * (1.0 - LABOR_SHARE),
        emergency_multiplier=1.0 / ratio,
    )
