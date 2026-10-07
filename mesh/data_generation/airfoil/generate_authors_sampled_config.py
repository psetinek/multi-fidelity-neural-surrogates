#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:  # noqa: D401 - PyYAML hook signature
        return True


def _format_re_tag(reynolds: float) -> str:
    # Keep compact naming style used in existing configs (e.g. 4.395e6, 4.08e6).
    return f"{reynolds / 1e6:.3f}".rstrip("0").rstrip(".")


def _format_aoa_tag(aoa_deg: float) -> str:
    # Example: -0.16 -> m0p16, 10.44 -> 10p44
    return f"{aoa_deg:.2f}".replace("-", "m").replace(".", "p")


def _naca_code_from_digits(digits: List[int]) -> str:
    if len(digits) == 3:
        return f"{digits[0]}{digits[1]}{digits[2]:02d}"
    if len(digits) == 4:
        return f"{digits[0]}{digits[1]}{digits[2]}{digits[3]:02d}"
    raise ValueError(f"Unsupported digits length {len(digits)} (expected 3 or 4).")


def _round_int_scalar(value: float) -> int:
    return int(np.round(value))


def _sample_cases(
    n_cases: int,
    seed: int,
    re_min_million: float,
    re_max_million: float,
    aoa_min: float,
    aoa_max: float,
    prefix: str,
) -> List[Dict[str, Any]]:
    if n_cases < 1:
        raise ValueError("n_cases must be >= 1")

    rng = np.random.default_rng(seed)
    n_four = n_cases // 2
    n_five = n_cases - n_four

    reynolds = rng.uniform(re_min_million, re_max_million, n_cases) * 1e6
    aoa = rng.uniform(aoa_min, aoa_max, n_cases)

    # 4-digit family: (M, P, XX)
    m = rng.uniform(0.0, 7.0, n_four)
    p4 = rng.uniform(0.0, 7.0, n_four)
    p4[p4 < 1.5] = 0.0
    xx4 = rng.uniform(5.0, 20.0, n_four)

    # 5-digit family: (L, P, Q, XX)
    l = rng.uniform(0.0, 4.0, n_five)
    p5 = rng.uniform(3.0, 8.0, n_five)
    q = rng.integers(0, 2, size=n_five)
    xx5 = rng.uniform(5.0, 20.0, n_five)

    design_space: List[Dict[str, Any]] = []
    for i in range(n_four):
        digits = [_round_int_scalar(m[i]), _round_int_scalar(p4[i]), _round_int_scalar(xx4[i])]
        design_space.append(
            {
                "reynolds": float(np.round(reynolds[i], 3)),
                "aoa_deg": float(np.round(aoa[i], 3)),
                "digits": digits,
            }
        )

    for i in range(n_five):
        digits = [
            _round_int_scalar(l[i]),
            _round_int_scalar(p5[i]),
            int(q[i]),
            _round_int_scalar(xx5[i]),
        ]
        design_space.append(
            {
                "reynolds": float(np.round(reynolds[i + n_four], 3)),
                "aoa_deg": float(np.round(aoa[i + n_four], 3)),
                "digits": digits,
            }
        )

    order = rng.permutation(n_cases)
    shuffled = [design_space[int(i)] for i in order]

    cases: List[Dict[str, Any]] = []
    for idx, case in enumerate(shuffled):
        code = _naca_code_from_digits(case["digits"])
        re_tag = _format_re_tag(case["reynolds"])
        aoa_tag = _format_aoa_tag(case["aoa_deg"])
        name = f"{prefix}{idx:03d}_naca{code}_re{re_tag}e6_aoa{aoa_tag}"
        cases.append(
            {
                "name": name,
                "reynolds": case["reynolds"],
                "aoa_deg": case["aoa_deg"],
                "digits": case["digits"],
            }
        )

    return cases


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate an airfoil multi-fidelity config using authors-like case sampling."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=here / "config_default.yaml",
        help="Template YAML to copy non-case settings from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output YAML path.",
    )
    parser.add_argument("--num-cases", type=int, default=10, help="Number of cases to sample.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--aoa-min", type=float, default=-5.0, help="Minimum AoA (deg).")
    parser.add_argument("--aoa-max", type=float, default=15.0, help="Maximum AoA (deg).")
    parser.add_argument(
        "--re-min-million",
        type=float,
        default=2.0,
        help="Minimum Reynolds number in millions (e.g. 2.0 -> 2e6).",
    )
    parser.add_argument(
        "--re-max-million",
        type=float,
        default=6.0,
        help="Maximum Reynolds number in millions (e.g. 6.0 -> 6e6).",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="sample",
        help="Case name prefix (e.g. sample, train, extreme).",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run_name override in output config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aoa_min > args.aoa_max:
        raise ValueError("--aoa-min must be <= --aoa-max")
    if args.re_min_million <= 0 or args.re_max_million <= 0:
        raise ValueError("--re-min-million/--re-max-million must be > 0")
    if args.re_min_million > args.re_max_million:
        raise ValueError("--re-min-million must be <= --re-max-million")

    with args.template.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg_out = copy.deepcopy(cfg)

    cases = _sample_cases(
        n_cases=int(args.num_cases),
        seed=int(args.seed),
        re_min_million=float(args.re_min_million),
        re_max_million=float(args.re_max_million),
        aoa_min=float(args.aoa_min),
        aoa_max=float(args.aoa_max),
        prefix=str(args.name_prefix),
    )

    cfg_out["cases"] = cases
    if args.run_name is not None:
        cfg_out["run_name"] = str(args.run_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.dump(cfg_out, handle, Dumper=_NoAliasDumper, sort_keys=False)

    print(
        f"Wrote {args.output} with {len(cases)} sampled cases "
        f"(seed={args.seed}, Re=[{args.re_min_million},{args.re_max_million}]e6, "
        f"AoA=[{args.aoa_min},{args.aoa_max}] deg)."
    )


if __name__ == "__main__":
    main()
