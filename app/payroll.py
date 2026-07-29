"""
Wage + revenue-proxy math on top of the real attendance/tracking data.
Hourly wage is a per-person config value (settable via API), hours worked comes
from the real attendance.working_hours_today / sightings_for data, and
"cash collected" is cups-detected * an assumed price-per-cup (explicitly a demo
proxy, not a POS integration).
"""
import json
import os
from datetime import date

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "payroll_config.json")
DEFAULT_HOURLY_WAGE = 15.0
DEFAULT_PRICE_PER_CUP = 30.0


def _load():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"wages": {}, "price_per_cup": DEFAULT_PRICE_PER_CUP}


def _save(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def set_wage(name: str, hourly_wage: float):
    cfg = _load()
    cfg["wages"][name] = hourly_wage
    _save(cfg)
    return cfg["wages"][name]


def get_wage(name: str) -> float:
    return _load()["wages"].get(name, DEFAULT_HOURLY_WAGE)


def set_price_per_cup(price: float):
    cfg = _load()
    cfg["price_per_cup"] = price
    _save(cfg)
    return price


def get_price_per_cup() -> float:
    return _load().get("price_per_cup", DEFAULT_PRICE_PER_CUP)


def wage_summary(name: str, working_hours_rows: list):
    """working_hours_rows: rows from attendance.working_hours_today() filtered to `name`,
    or any list of {span_minutes, sightings} dicts for the days present."""
    hourly = get_wage(name)
    total_minutes = sum(r["span_minutes"] for r in working_hours_rows)
    total_hours = round(total_minutes / 60, 2)
    pay = round(total_hours * hourly, 2)
    return {
        "name": name,
        "hourly_wage": hourly,
        "total_hours": total_hours,
        "estimated_pay": pay,
        "days_counted": len(working_hours_rows),
    }


def cash_collected(cup_count: int):
    price = get_price_per_cup()
    return {
        "cup_count": cup_count,
        "price_per_cup": price,
        "estimated_cash": round(cup_count * price, 2),
        "note": "Illustrative: cup_count is a detection-count proxy, price_per_cup is a "
                "configurable assumption, not a POS integration.",
    }
