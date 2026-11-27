# -*- coding: utf-8 -*-
# @Time : 2026/11/26 下午9:00
# @Author : Wang Jun
# @Email: wj_xd@foxmail.com

"""Run bounded acceptance examples without replacing product execution."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from examples.acceptance.cases import (  # noqa: E402
    CASES,
    GROUPS,
    build_plan,
    validate_environment,
)
from examples.acceptance.checks import (  # noqa: E402
    check_inference,
    check_training,
    verify_stream,
)
from examples.acceptance.preflight import model_layout_blockers  # noqa: E402
from examples.acceptance.processes import ProcessCleanupError, run_process  # noqa: E402
from examples.acceptance.storage import required_free, storage_plan  # noqa: E402


class BlockedError(RuntimeError):
    """A prerequisite stopped execution without weakening the selected case."""


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def distribution_identity():
    try:
        distribution = importlib.metadata.distribution("parascale")
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "record_sha256": None}
    record = distribution.read_text("RECORD")
    return {"version": distribution.version,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None}


def load_environment(path):
    path = Path(path).resolve()
    env = validate_environment(json.loads(path.read_text(encoding="utf-8")))
    env["assets"] = {key: str((path.parent / value).resolve()) if value else ""
                     for key, value in env["assets"].items()}
    return env


def validate_output(output, repo, assets):
    """Never overwrite a run or write results into the checkout/input assets."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output must be a new directory")
    for protected in [Path(repo).resolve(), *(Path(p).resolve() for p in assets)]:
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"output overlaps repository or input assets: {protected}")
    return output


def asset_identity(path):
    path = Path(path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = {}
    for file in files:
        if file.is_symlink() and file.is_dir():
            raise ValueError(f"directory symlink cannot be inventoried: {file}")
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            result[file.relative_to(path).as_posix() if path.is_dir() else file.name] = digest.hexdigest()
    if not result:
        raise ValueError(f"empty input asset: {path}")
    return result


def prerequisites(case, env, train, infer):
    reasons = []
    if case.blocked:
        reasons.append(case.blocked)
    if case.gpus > len(env["gpu_ids"]):
        reasons.append(f"requires {case.gpus} GPUs; configured {len(env['gpu_ids'])}; four-GPU resource cap")
    for name in case.assets:
        value = env["assets"].get(name)
        path = Path(value) if value else None
        valid = path is not None and (path.is_file() if name.endswith("prompts") else path.is_dir())
        if not valid:
            reasons.append(f"missing local asset: {name}")
    if case.kind in ("installation", "parallel") and not train:
        reasons.append("parascale-train not installed/on PATH")
    if case.kind in ("installation", "transformer", "diffusion") and not infer:
        reasons.append("parascale-infer not installed/on PATH")
    return reasons


def prepare_case(case, env, directory, train, infer):
    reasons = prerequisites(case, env, train, infer)
    reasons.extend(model_layout_blockers(case, env))
    try:
        plan = build_plan(case, env, directory, train or "parascale-train", infer or "parascale-infer")
    except ValueError as exc:
        plan = {"commands": [], "configs": {}, "acceptance_gaps": []}
        reasons.append(f"case configuration blocked: {exc}")
    return plan, reasons


def _verify(case, directory, steps):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8"))

    if case.kind == "installation":
        versions = []
        for index, name in enumerate(("parascale-train", "parascale-infer")):
            version = (directory / f"{index * 2:02d}.stdout.log").read_text(encoding="utf-8").strip().split()
            help_text = (directory / f"{index * 2 + 1:02d}.stdout.log").read_text(encoding="utf-8")
            if len(version) != 2 or version[0] != name or name not in help_text:
                raise ValueError("unexpected installed entrypoint identity")
            versions.append(version[1])
        if versions[0] != versions[1]:
            raise ValueError("product versions differ")
        return {"version": versions[0]}
    if case.kind == "parallel":
        train, resume = read("train-result.json"), read("resume-result.json")
        metrics = check_training(train, resume, steps)
        for payload in (train, resume):
            checkpoint = Path(payload["checkpoint"]["path"]).resolve()
            if not checkpoint.is_relative_to(directory) or not checkpoint.is_dir() or not any(checkpoint.iterdir()):
                raise ValueError("checkpoint evidence missing or outside case output")
        return metrics
    if case.kind in ("transformer", "diffusion"):
        payload = read("inference-result.json")
        metrics = check_inference(payload, case.kind)
        verify_stream(payload, case.kind, directory)
        return metrics
    return {}


def execute_suite(raw, output, case_ids, *, mode="acceptance", process=None, tools=None):
    """Run selected functional probes; missing formal gates remain BLOCKED."""
    env = validate_environment(raw)
    if mode not in ("functional", "acceptance") or not case_ids or any(key not in CASES for key in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("select known distinct cases and a valid mode")
    output = validate_output(output, ROOT, [value for value in env["assets"].values() if value])
    train, infer = tools or (shutil.which("parascale-train"), shutil.which("parascale-infer"))
    missing_psutil = process is None and importlib.util.find_spec("psutil") is None
    process = process or run_process
    plans, blocked = {}, {}
    for key in case_ids:
        case = CASES[key]
        plans[key], blocked[key] = prepare_case(case, env, output / key, train, infer)
        if missing_psutil:
            blocked[key].append("install examples/acceptance/requirements.txt before execution")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema_version": 1, "mode": mode, "environment": env, "plans": plans,
                "case_ids": case_ids, "asset_sha256": {}, "preflight": blocked,
                "python": sys.version, "product_distribution": distribution_identity(), "source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(output / "protocol.json", protocol)
    needed = {name for key in case_ids if not blocked[key] for name in CASES[key].assets}
    for name in sorted(needed):
        try:
            protocol["asset_sha256"][name] = asset_identity(env["assets"][name])
        except (OSError, ValueError) as exc:
            for key in case_ids:
                if name in CASES[key].assets:
                    blocked[key].append(str(exc))
    available = shutil.disk_usage(output).free
    protocol["initial_free_bytes"] = available
    for key in case_ids:
        if blocked[key]:
            continue
        try:
            allocation = storage_plan(CASES[key], env)
            plans[key]["storage"] = allocation
            if required_free(allocation, 0) > available:
                blocked[key].append("insufficient disk space for cumulative retained outputs and reserve")
            else:
                available -= allocation["retained_bytes"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blocked[key].append(f"cannot estimate disk capacity: {exc}")
    # Freeze all executable inputs before the first subprocess starts.
    write_json(output / "protocol.json", protocol)
    report = {"schema_version": 1, "scope": mode, "status": "BLOCKED", "full_acceptance_passed": False,
              "performance_claim_eligible": False, "protocol": str(output / "protocol.json"),
              "unselected_cases": sorted(set(CASES) - set(case_ids)), "cases": []}
    stop_reason = None
    for key in case_ids:
        case, plan, directory = CASES[key], plans[key], output / key
        row = {"id": key, "requirements": list(case.requirements), "status": "BLOCKED",
               "functional_status": "BLOCKED", "reasons": list(blocked[key]),
               "acceptance_gaps": plan["acceptance_gaps"], "commands": [], "metrics": {}}
        report["cases"].append(row)
        if stop_reason:
            row["reasons"].append(stop_reason)
        elif not row["reasons"]:
            directory.mkdir()
            for name, config in plan["configs"].items():
                write_json(directory / name, config)
            child_env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, env["gpu_ids"][:case.gpus])),
                         "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            try:
                for index, command in enumerate(plan["commands"]):
                    free = shutil.disk_usage(output).free
                    required = required_free(plan["storage"], index if case.kind == "parallel" else 0)
                    if free < required:
                        raise BlockedError(f"insufficient disk space before command: free={free}, required={required}")
                    result = process(command, directory, child_env, env["timeout_seconds"], directory / f"{index:02d}")
                    row["commands"].append({"argv": command, "free_bytes_before": free, "required_free_bytes": required, **result})
                    if result["returncode"] != 0 or result["timed_out"]:
                        raise ValueError("command failed or timed out; see retained logs")
                row["metrics"] = _verify(case, directory, env["budgets"]["train_steps"])
                row["functional_status"] = "PASS"
                row["status"] = "BLOCKED" if mode == "acceptance" and plan["acceptance_gaps"] else "PASS"
                if row["status"] == "BLOCKED":
                    row["reasons"].extend(plan["acceptance_gaps"])
            except KeyboardInterrupt:
                stop_reason = "suite interrupted before execution"
                row.update(status="FAIL", functional_status="FAIL", reasons=["interrupted; launched process group terminated"])
            except ProcessCleanupError as exc:
                stop_reason = "process cleanup failed; remaining cases were not launched"
                row.update(status="FAIL", functional_status="FAIL", reasons=[str(exc)])
            except BlockedError as exc:
                row["reasons"].append(str(exc))
            except Exception as exc:
                row.update(status="FAIL", functional_status="FAIL", reasons=[str(exc)])
        write_json(output / "summary.json", report)
    statuses = {row["status"] for row in report["cases"]}
    report["status"] = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
    write_json(output / "summary.json", report)
    lines = ["# ParaScale Acceptance Examples", "", f"Scope: {mode}; status: {report['status']}",
             "", "Full production acceptance: NOT PASSED. Performance claims: NOT QUALIFIED.",
             "", "| Case | Functional | Scope Result |", "| --- | --- | --- |"]
    lines += [f"| {row['id']} | {row['functional_status']} | {row['status']} |" for row in report["cases"]]
    lines += ["", "See summary.json for blockers, metrics and logs; protocol.json freezes inputs."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print commands/blockers without launching or writing outputs")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--group", action="append", choices=GROUPS)
    parser.add_argument("--mode", choices=("functional", "acceptance"), default="acceptance")
    args = parser.parse_args(argv)
    if args.list:
        for case in CASES.values():
            print(f"{case.id}: {','.join(case.requirements)}; GPUs={case.gpus}; {'BLOCKED: ' + case.blocked if case.blocked else 'functional probe'}")
        return 0
    if not args.config or not args.output:
        parser.error("--config and --output are required")
    if args.case and args.group:
        parser.error("choose --case or --group, not both")
    selected = args.case or [key for key, case in CASES.items() if not args.group or case.group in args.group]
    try:
        env = load_environment(args.config)
        if args.plan:
            output = validate_output(args.output, ROOT, [value for value in env["assets"].values() if value])
            train, infer = shutil.which("parascale-train"), shutil.which("parascale-infer")
            result = {}
            for key in selected:
                plan, reasons = prepare_case(CASES[key], env, output / key, train, infer)
                result[key] = {**plan, "blockers": reasons}
            print(json.dumps(result, indent=2))
            return 0
        report = execute_suite(env, args.output, selected, mode=args.mode)
        print(json.dumps({"scope": args.mode, "status": report["status"], "summary": str(args.output / "summary.json")}, indent=2))
        return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[report["status"]]
    except (OSError, ValueError) as exc:
        print(f"acceptance configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
