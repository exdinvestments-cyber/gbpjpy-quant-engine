"""Human-readable risk audit: why this risk, this volume, this pip value, these rates, this binding constraint."""

from __future__ import annotations

from ..reason_codes import describe


def explain_risk_decision(dec: dict) -> str:
    a, au = dec.get("approved_trade"), dec.get("audit", {})
    lines = [f"RISK DECISION {dec['decision']} for {dec['trade_proposal_id']} at {dec['timestamp']}",
             f"  global state {dec['global_risk_state']}  binding constraint {dec.get('binding_constraint')}"]
    if dec.get("rejection_category"):
        lines.append(f"  REJECTED [{dec['rejection_category']}]: {dec['rejection_reason']}")
    rb = au.get("risk_budget")
    if rb:
        lines += ["", "WHY WAS THIS AMOUNT OF RISK PERMITTED?",
                  f"  sizing capital {rb['sizing_capital']} ({rb['basis']}); strictest cap {rb['binding_constraint']} -> "
                  f"permitted {rb['permitted_risk']}"]
        lines += [f"    {k:<24} {v}" for k, v in rb["caps"].items()]
    if au.get("conversion"):
        cv = au["conversion"]
        lines += ["WHAT CONVERSION RATES WERE USED?", f"  {cv['from']}->{cv['to']} factor {cv['factor']}"]
        lines += [f"    {r['pair']} {r['rate']} @ {r['timestamp']} ({r['source']})" for r in cv["rates"]]
    if a:
        lines += ["WHAT PIP VALUE / STOP DISTANCE / VOLUME?",
                  f"  pip value per 1.0 volume {a['pip_value_per_volume_unit']} {a['account_currency']} "
                  f"({a['pip_value_quote_per_volume_unit']} JPY); stop distance {a['stop_distance_pips']} pips "
                  f"(+ allowances = {a['risk_distance_pips_incl_allowances']})",
                  f"  theoretical volume {a['theoretical_volume']} -> approved {a['approved_volume']} (rounded DOWN)",
                  f"  actual risk {a['actual_risk_currency']} {a['account_currency']} = {a['actual_risk_percent']}% "
                  f"(planned {a['planned_risk_currency']}; not a guaranteed maximum loss - gap risk {a['gap_risk_status']})"]
    if au.get("periods"):
        lines.append("DAILY / WEEKLY / MONTHLY BUDGET (before)")
        lines += [f"  {k}: {v}" for k, v in au["periods"].items()]
    if au.get("decision"):
        lines.append(f"  after: daily {au['decision']['daily_after']}  aggregate {au['decision']['aggregate_before']} -> "
                     f"{au['decision']['aggregate_after']}")
    if au.get("drawdown"):
        lines.append(f"DRAWDOWN STATE {au['drawdown']['state']} ({au['drawdown']['equity_drawdown_pct']}% from HWM "
                     f"{au['drawdown']['hwm_equity']})")
    if au.get("margin"):
        lines.append(f"MARGIN {au['margin']}")
    lines += ["", "REASON CODES"] + [f"  {c:<34} {describe(c)}" for c in dec["reason_codes"]]
    lines.append("\nnot an order - risk approval only")
    return "\n".join(lines)
