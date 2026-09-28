"""Human-readable trade-construction audit: why this stop, why this target, what is unknown."""

from __future__ import annotations

from ..reason_codes import describe


def _f(v, nd=2):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def explain_trade(result, trade_proposal_id: str) -> str:
    rec = result.construction(trade_proposal_id)
    p = rec.proposal
    lines = [f"GBPJPY {rec.direction} trade construction {rec.trade_proposal_id}  (entry {rec.entry_candidate_id})",
             f"  decision {rec.decision}  state {rec.state}" + (f"  [{rec.category}] {rec.reason}" if rec.category else ""),
             "  STATE TRANSITIONS"]
    lines += [f"    {t[1]}  {t[2]} -> {t[3]}  ({t[4]})" for t in rec.transitions]
    if not p:
        return "\n".join(lines + ["  (no geometry was built)"])
    st = p["stop_detail"]
    lines += [
        "", "WHY THIS STOP?",
        f"  {p['stop_reference_type']} at {_f(p['stop_reference_price'], 3)}: {p['structural_reason']}",
        f"  raw invalidation {_f(p['raw_invalidation_price'], 3)} + buffer {p['stop_buffer_pips']} pips -> stop "
        f"{_f(p['proposed_stop_price'], 3)}  ({p['stop_distance_pips']} pips, {_f(p['stop_distance_atr'])} ATR, "
        f"{p['volatility_adequacy']}, noise risk {_f(p['stop_noise_risk_score'], 1)}, quality {_f(p['stop_quality_score'], 1)})",
        "WHY IS THE STOP NOT CLOSER?", f"  {st['why_not_closer']}",
        "", "WHY THIS TARGET?", f"  {p['primary_target_reason']}",
    ]
    for t in p["candidate_targets"]:
        lines.append(f"  {t['label']} {_f(t['price'], 3)} {t['type']} ({t['timeframe']}, strength {_f(t['strength'], 0)}) "
                     f"{t['distance_pips']} pips  gross R {_f(t['gross_R'])}  net R {_f(t['estimated_net_R'])}  "
                     f"reachability {_f(t['target_reachability_score'], 0)}  barriers before {t['barriers_before_target']}")
    pt = p["primary_target"]
    if pt:
        lines += ["WHY IS THE TARGET REALISTIC?",
                  f"  placed {pt['placement']}; reachability {_f(pt['target_reachability_score'], 0)} from entry-time context",
                  "WHAT BARRIERS EXIST?",
                  f"  {pt['path']['barrier_count']} before the primary target (strong {pt['path']['strong_barrier_count']}, "
                  f"density {_f(pt['path']['barrier_density_score'], 0)}, nearest {pt['path']['nearest_barrier']})"]
    lines += ["", f"GROSS R {_f(p['gross_R'])}   ESTIMATED NET R {_f(p['estimated_net_R'])}   (minimum {p['minimum_net_R_required']})",
              "WHAT IS UNKNOWN?"]
    for k, v in p["cost_assumptions"]["components"].items():
        lines.append(f"  {k:<22} {v['status']:<16} {v.get('pips')}")
    lines += [f"  broker stop constraints {p['broker_constraints_status']}; news {p['news_status']}",
              "", f"WHY {rec.decision}?  {rec.reason}", "REASON CODES"]
    lines += [f"  {c:<40} {describe(c)}" for c in rec.reason_codes]
    lines.append("\nnot an order - no volume, lot size or monetary risk exists")
    return "\n".join(lines)
