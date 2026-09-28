"""Human-readable execution audit: why it qualified, why risk approved it, what execution saw and did."""

from __future__ import annotations

from ..reason_codes import describe


def explain_execution(audit: dict) -> str:
    lines = [f"EXECUTION AUDIT - final state {audit['final_state']}",
             "WHY THE STRATEGY QUALIFIED IT", f"  {audit['why_strategy_qualified']}",
             "WHY RISK APPROVED IT", f"  {audit['why_risk_approved']}",
             f"REQUESTED (reference, not a guaranteed fill) {audit['requested_price']}"]
    g = audit.get("execution_conditions")
    if g:
        lines += ["FINAL GATE", f"  {g.get('outcome')}: {g.get('reason')}"]
        lines += [f"    {c:<32} {describe(c)}" for c in g.get("reason_codes", [])]
        for k in ("executable_price", "spread_pips", "validated_volume", "conversion_factor"):
            if k in g:
                lines.append(f"    {k:<32} {g[k]}")
    lines.append(f"ADAPTER ACKNOWLEDGEMENT {audit.get('adapter_acknowledgement')}")
    for f in audit.get("fills", []):
        led = f.get("ledger") or {}
        lines.append(f"  fill {f['volume']} @ {f['price']} ({f['at']}) slippage {led.get('slippage_pips')} pips "
                     f"(positive = adverse) commission {f['commission']}")
    if audit.get("realised"):
        lines.append(f"REALISED {audit['realised']}")
    lines.append(f"PROTECTION {audit.get('protection')}")
    if audit.get("errors_and_retries"):
        lines.append("ERRORS / RETRIES / DEFERRALS")
        lines += [f"  {e['at']} {e['type']} {e['payload']}" for e in audit["errors_and_retries"]]
    lines.append("LIFECYCLE")
    lines += [f"  {h['at']} {h['from']} -> {h['to']}: {h['reason']}" for h in audit["lifecycle"]]
    return "\n".join(lines)
