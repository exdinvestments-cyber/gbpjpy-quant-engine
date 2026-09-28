"""Command-line entry point (research use only; no trading functionality).

Examples
--------
  python -m gbpjpy_engine run --csv gbpjpy_h4.csv --tz UTC --out output/features.parquet --log output/eval.jsonl
  python -m gbpjpy_engine inspect --csv gbpjpy_h4.csv --tz UTC --timestamp 2024-03-01T08:00:00Z
  python -m gbpjpy_engine synthetic --scenario clean_uptrend --out output/uptrend.csv
  python -m gbpjpy_engine config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import describe_config, load_config
from .data.model import load_csv
from .engine import H4MarketIntelligenceEngine
from .research import explain, explain_permission
from .synthetic import SCENARIOS, generate_scenario


def _load(args):
    if args.scenario:
        return generate_scenario(args.scenario).bars
    if not args.csv:
        raise SystemExit("provide --csv or --scenario")
    return load_csv(args.csv, assume_timezone=args.tz)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gbpjpy-h4", description="GBPJPY H4 Market Intelligence Engine (Phase 1A)")
    p.add_argument("--log-level", default="WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--csv", help="CSV of H4 bars (timestamp=bar open, open, high, low, close[, volume, spread])")
        sp.add_argument("--tz", default=None, help="timezone of NAIVE timestamps (required if timestamps are naive)")
        sp.add_argument("--scenario", choices=SCENARIOS, help="use a synthetic scenario instead of a CSV")
        sp.add_argument("--config", help="YAML/JSON config override file")

    r = sub.add_parser("run", help="evaluate every closed H4 bar and export results")
    common(r)
    r.add_argument("--out", help="feature table output (.parquet/.csv/.jsonl)")
    r.add_argument("--log", help="structured JSONL evaluation log output")
    r.add_argument("--context-out", help="Phase 1B context/permission table output (.parquet/.csv/.jsonl)")

    i = sub.add_parser("inspect", help="explain the evaluation of one H4 bar")
    common(i)
    i.add_argument("--timestamp", help="bar OPEN time (tz-aware ISO). Default: latest bar")
    i.add_argument("--json", action="store_true", help="print the full snapshot as JSON")
    i.add_argument("--why", action="store_true", help="only explain the directional permission")

    s = sub.add_parser("synthetic", help="write a synthetic scenario CSV (engineering tests only)")
    s.add_argument("--scenario", choices=SCENARIOS, required=True)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)

    sub.add_parser("config", help="print all parameters with their documented purpose")

    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.cmd == "config":
        for row in describe_config():
            print(f"{row['section']}.{row['name']} = {row['value']!r}\n    {row['doc']}")
        return 0
    if args.cmd == "synthetic":
        generate_scenario(args.scenario, seed=args.seed).bars.to_csv(args.out, index=False)
        return 0

    engine = H4MarketIntelligenceEngine(load_config(args.config))
    result = engine.run(_load(args))
    if args.cmd == "run":
        if args.out:
            result.export_features(args.out)
        if args.log:
            result.write_evaluation_log(args.log)
        if args.context_out:
            result.export_context(args.context_out)
        print(json.dumps({"bars": len(result.features), "data_quality": result.report.summary(),
                          "regimes": result.features["regime"].value_counts().to_dict(),
                          "bias": result.features["h4_bias"].value_counts().to_dict(),
                          "directional_permission": result.context.frame["directional_permission"].value_counts().to_dict()},
                         indent=2, default=str))
        return 0
    snap = result.latest() if not args.timestamp else result.snapshot(args.timestamp)
    if args.why:
        print(explain_permission(result, args.timestamp))
        return 0
    print(snap.to_json() if args.json else explain(snap))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
