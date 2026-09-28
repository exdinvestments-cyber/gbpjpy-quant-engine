"""Human-readable audit of H1 setups ("why did this setup qualify / fail?")."""

from __future__ import annotations

from ..reason_codes import describe


def _fmt(v, nd=1):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def explain_h1_setup(result, setup_id: str) -> str:
    info = result.explain_setup(setup_id)
    q = info.get("qualification_explanation") or info.get("end_explanation") or {}
    lines = [f"GBPJPY {info['side'].upper()} {info['family']}  setup {info['setup_id']}",
             f"  created {info['created_at']}  state {info['state']}"
             + (f"  ended: {info['end_reason']}" if info.get("end_reason") else "")]
    for t in info["transitions"]:
        lines.append(f"    {t['at']}  {t['from']} -> {t['to']}  ({t['reason']})")
    if info.get("qualified"):
        qs = info["qualified"]
        lines += [
            "", "WHY IT QUALIFIED (not a trade - no entry/stop/target/size exists)",
            f"  H4: {qs['h4_permission']} (confidence {_fmt(qs['h4_permission_confidence'])}, "
            f"context score {_fmt(qs['h4_context_score'])})",
            f"  H1 structure: primary {qs['h1_primary_structure']}, immediate {qs['h1_immediate_structure']}",
            f"  pullback: {qs['pullback_state']} (quality {_fmt(qs['pullback_quality'])})  location {_fmt(qs['location_score'])}",
            f"  trigger: {qs['trigger']['source']} {_fmt(qs['trigger']['score'])}  displacement {_fmt(qs['displacement_score'])} "
            f"transition {_fmt(qs['transition_score'])} rejection {_fmt(qs['rejection_score'])}",
            f"  room {_fmt(qs['room_score'])}  chop {_fmt(qs['chop_score'])}  conflict {_fmt(qs['conflict_score'])}",
            f"  setup score {_fmt(qs['setup_score'])}  confidence {_fmt(qs['setup_confidence'])} "
            "(evidence consistency, not a win probability)",
            f"  invalidation reference {_fmt(qs['invalidation_reference'], 3)} (structural premise, not a stop)",
        ]
    lines += ["", "EVIDENCE / REASON CODES"] + [f"  {c:<48} {describe(c)}" for c in q.get("reason_codes", [])]
    if q.get("missing_requirements"):
        lines += ["", "UNMET REQUIREMENTS"] + [f"  {m}" for m in q["missing_requirements"]]
    return "\n".join(lines)


def explain_h1_bar(result, i: int) -> str:
    row = result.frame.iloc[i]
    lines = [f"H1 bar {row['timestamp']} (known at {row['available_at']})",
             f"  H4 context: bar {row.get('h4_timestamp')} status {row.get('h4_context_status')} "
             f"permission {row.get('h4_permission')}",
             f"  alignment {row.get('alignment_state')}  blockers {list(row.get('h1_blockers') or [])}"]
    for side in ("long", "short"):
        e = result.explain(i, side)
        lines.append(f"  {side.upper():5} state {e.get('state')} family {e.get('setup_family')} score "
                     f"{_fmt(e.get('setup_score'))} confidence {_fmt(e.get('setup_confidence'))} "
                     f"actionable {e.get('actionable_setup_permission')}")
        for m in e.get("missing_requirements") or []:
            lines.append(f"        unmet: {m}")
    return "\n".join(lines)
