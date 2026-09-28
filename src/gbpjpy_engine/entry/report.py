"""Human-readable entry audit ("how did this candidate evolve, and why?")."""

from __future__ import annotations

from ..reason_codes import describe


def _fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def explain_entry_candidate(result, entry_candidate_id: str) -> str:
    info = result.explain_candidate(entry_candidate_id)
    lines = [f"GBPJPY {info['side'].upper()} entry candidate {info['entry_candidate_id']}",
             f"  setup {info['setup_id']} ({info['setup_family']}) qualified {info['qualified_at']}",
             f"  final state {info['state']}" + (f"  ({info['end_reason']})" if info.get("end_reason") else ""),
             "", "AUDIT TRAIL"]
    for a in info["audit"]:
        detail = "" if a["detail"] in (None, "", {}) else f"  [{a['detail']}]"
        lines.append(f"  {a['at']}  {a['phase']:<5}  {a['step']}{detail}")
    lines += ["", "STATE TRANSITIONS"]
    for t in info["transitions"]:
        lines.append(f"  {t['at']}  {t['phase']:<5}  {t['from']} -> {t['to']}  ({t['reason']})")
    if info.get("accepted"):
        acc = info["accepted"]
        lines += ["", "ACCEPTED ENTRY CANDIDATE (not an order - no size, stop, target or ticket exists)",
                  f"  signal price {_fmt(acc['signal_price'], 3)} ({acc['signal_price_basis']} close of the confirmation bar)",
                  f"  executable reference {_fmt(acc['executable_reference_price'], 3)} on the {acc['execution_side']} "
                  f"(spread {acc['spread_status']} {_fmt(acc['spread_pips'])} pips, slippage {acc['slippage_status']})",
                  f"  chase {_fmt(acc['chase_risk_score'])}  extension {_fmt(acc['extension_score'])}  freshness "
                  f"{_fmt(acc['freshness_score'])}  room {_fmt(acc['remaining_room_score'])}  conflict "
                  f"{_fmt(acc['entry_conflict_score'])}  quality {_fmt(acc['entry_quality_score'])}"]
        lines += ["", "REASON CODES"] + [f"  {c:<44} {describe(c)}" for c in acc["reason_codes"]]
    elif info["opportunities"]:
        last = info["opportunities"][-1]
        lines += ["", f"LAST DECISION: {last.get('decision')} - {last.get('reason')}"]
        lines += [f"  {c['check']:<22} {c['result']:<7} {c['value']}" for c in last.get("checks", [])]
    return "\n".join(lines)
