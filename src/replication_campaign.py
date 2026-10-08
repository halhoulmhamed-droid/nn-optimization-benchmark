"""Phase 3B sequential integration, supplied-label training, shared physical cache.

No physical oracle is imported here. Adam and checkpoint contracts are those
of phase3A. Real callbacks and M0 are loaded only by the explicit script mode.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time

from src import replication_training as r
from src import neural_surrogate as n
from src import surrogate_decision as sd
from src.pilot_training import saturation_diagnostic  # pure, no oracle import

SCHEMA = 1
SAVE_EVERY = 100
METRIC_FIELDS = (
    "seed", "condition", "step", "status", "missing_reason", "parameter_id",
    "rmse_value", "rmse_derivative", "rmse_value_normalized", "rmse_derivative_normalized",
    "max_abs_value_error_discrete", "max_abs_derivative_error_discrete",
    "mean_D_interior", "rounded_saturation_fraction", "near_saturation_fraction",
    "boundary_abs_value_error_a", "boundary_abs_value_error_b",
    "boundary_abs_derivative_error_a", "boundary_abs_derivative_error_b",
    "training_loss", "n_value_labels", "n_derivative_labels", "C_label_q1_1",
)
DECISION_FIELDS = (
    "seed", "condition", "step", "task", "task_class", "lambda_value", "status",
    "missing_reason", "p_hat", "p_ref", "physical_branch", "candidate_origin",
    "predicted_objective", "physical_objective", "reference_objective", "D",
    "physical_gradient", "neural_gradient", "physical_kkt", "neural_kkt",
    "alpha_grid_float", "beta_ref_conditional", "branch_mismatch",
    "D_negative", "numerical_status",
)


class IntegrityError(ValueError):
    """Recipe/code/context conflict: stop the campaign; do not change algorithms."""


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def load_record(path, expected_context=None):
    record = r.read_json(path)
    digest = record.pop("record_sha256")
    if r.identity(record) != digest:
        raise IntegrityError(f"record integrity mismatch: {path}")
    if expected_context is not None and record["context_id"] != expected_context:
        raise IntegrityError(f"record context collision: {path}")
    return record


def atomic_record(path, body, *, immutable=False):
    """Single-writer, same-directory temporary; no universal crash guarantee."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = r._copy_json(body)
    payload = dict(body, record_sha256=r.identity(body))
    before_hash = file_hash(path) if path.exists() else None
    if path.exists():
        old = load_record(path)
        if old.get("context_id") != body.get("context_id"):
            raise IntegrityError(f"refuse context collision: {path}")
        if r.canonical_bytes(old) == r.canonical_bytes(body):
            return False
        if immutable:
            raise IntegrityError(f"immutable output differs: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name+".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            import json
            handle.write(json.dumps(payload, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if r.canonical_bytes(load_record(temporary)) != r.canonical_bytes(body):
            raise IntegrityError("temporary record failed round-trip")
        if (file_hash(path) if path.exists() else None) != before_hash:
            raise IntegrityError("destination changed during sequential write")
        os.replace(temporary, path)
        temporary = None
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class PhysicalCache:
    """Exact keys (quantity,p.hex); origin is retained, never approximate merging."""
    def __init__(self, oracle_id, *, value_call=None, derivative_call=None,
                 scope="REAL_THERMAL", entries=None, counts=None, used=None):
        r._digest(oracle_id, "oracle id")
        self.oracle_id, self.scope = oracle_id, scope
        self.callbacks = {"T": value_call, "T_prime": derivative_call}
        self.entries = {}
        self.counts = counts or {
            q: {"requests": 0, "hits": 0, "callback_attempts": 0,
                "new_calls": 0, "archived_added": 0} for q in self.callbacks}
        self.used = {key: set(value) for key, value in (used or {}).items()}
        for key, entry in (entries or {}).items():
            self._validate_entry(key, entry)
            self.entries[key] = r._copy_json(entry)
        for q in self.callbacks:
            for name, value in self.counts[q].items():
                r.integer(value, f"{q}.{name}", 10**12)
            if self.counts[q]["requests"] != self.counts[q]["hits"] + self.counts[q]["callback_attempts"]:
                raise IntegrityError("cache request counts inconsistent")

    @staticmethod
    def key(quantity, p):
        if quantity not in ("T", "T_prime"):
            raise ValueError("unknown physical quantity")
        return quantity + ":" + n._validate_p(p).hex()

    def _validate_entry(self, key, entry):
        p = n._validate_p(entry["p"])
        if entry["p_hex"] != p.hex() or key != self.key(entry["quantity"], p):
            raise IntegrityError("cache point/quantity identity mismatch")
        if entry["oracle_id"] != self.oracle_id:
            raise IntegrityError("cache oracle identity mismatch")
        n._finite_scalar(entry["value"], "cache value")
        if entry["origin"] not in ("ARCHIVED", "NEW_CALL", "IMPORT_PREPARATION", "TEST_FIXTURE"):
            raise IntegrityError("unknown label origin")
        if not isinstance(entry["provenance"], dict) or not entry["provenance"]:
            raise IntegrityError("cache label provenance missing")

    def seed(self, quantity, p, value, *, origin, provenance):
        p, value = n._validate_p(p), n._finite_scalar(value, "label")
        key = self.key(quantity, p)
        candidate = dict(quantity=quantity, p=p, p_hex=p.hex(), value=value,
                         origin=origin, oracle_id=self.oracle_id, provenance=provenance)
        self._validate_entry(key, candidate)
        if key in self.entries:
            if self.entries[key]["value"].hex() != value.hex():
                raise IntegrityError("conflicting exact-site label")
            return False
        self.entries[key] = r._copy_json(candidate)
        if origin == "ARCHIVED":
            self.counts[quantity]["archived_added"] += 1
        return True

    def get(self, quantity, p, *, role):
        key = self.key(quantity, p)
        self.counts[quantity]["requests"] += 1
        self.used.setdefault(role, set()).add(key)
        if key in self.entries:
            self.counts[quantity]["hits"] += 1
            return self.entries[key]["value"]
        self.counts[quantity]["callback_attempts"] += 1
        function = self.callbacks[quantity]
        if not callable(function):
            raise IntegrityError("missing physical callback for an uncached label")
        value = n._finite_scalar(function(n._validate_p(p)), "physical callback")
        self.seed(quantity, p, value, origin=("TEST_FIXTURE" if self.scope == "TEST_FIXTURE"
                                             else "NEW_CALL"),
                  provenance={"oracle_id": self.oracle_id, "method": "direct callback"})
        self.counts[quantity]["new_calls"] += 1
        return value

    def fingerprint(self, method, points):
        # Only actual job targets; not the whole cache or its mutable counters.
        quantities = ("T",) if method == "M1" else ("T", "T_prime")
        subset = []
        for p in points:
            for q in quantities:
                key = self.key(q, p)
                if key not in self.entries:
                    raise IntegrityError("required used label is missing")
                entry = self.entries[key]
                self._validate_entry(key, entry)
                subset.append({k: entry[k] for k in
                               ("quantity", "p_hex", "value", "oracle_id", "provenance")})
        return r.identity(subset)

    def payload(self):
        return r._copy_json(dict(schema_version=SCHEMA, kind="physical_cache", context_id=self.oracle_id,
                    scope=self.scope, entries=self.entries, counts=self.counts,
                    used={role: sorted(keys) for role, keys in self.used.items()}))

    def save(self, path):
        return atomic_record(path, self.payload())

    @classmethod
    def load(cls, path, oracle_id, **callbacks):
        data = load_record(path, oracle_id)
        return cls(oracle_id, scope=data["scope"], entries=data["entries"],
                   counts=data["counts"], used=data["used"], **callbacks)

    def report(self):
        by_quantity = {}
        all_used = set().union(*self.used.values()) if self.used else set()
        for q, counts in self.counts.items():
            requested = {key for key in all_used if key.startswith(q+":")}
            by_quantity[q] = dict(counts,
                unique_entries=sum(e["quantity"] == q for e in self.entries.values()),
                unique_requested_sites=len(requested),
                archived_unique_used=sum(self.entries[k]["origin"] == "ARCHIVED"
                                          for k in requested if k in self.entries))
        return dict(by_quantity=by_quantity,
                    role_unique_requests={role: len(keys) for role, keys in self.used.items()},
                    exact_site_key="quantity:p.hex(); no approximate merging")


def acquire_datasets(plan, cache):
    """Only supplied targets enter adapters; diagnostics are kept in separate objects."""
    p16, p32 = r.midpoint_sites(16), r.midpoint_sites(32)
    diagnostics = tuple(n.A + (j + 0.5) * (n.B - n.A) / 257 for j in range(257))
    if len(set(diagnostics)) != 257 or set(diagnostics) & (set(p16) | set(p32)):
        raise IntegrityError("diagnostic sites are not strictly held out")
    training = {}
    for condition in plan["replication"]["conditions"]:
        points = p16 if condition["training_sites"] == 16 else p32
        values = tuple(cache.get("T", p, role="training") for p in points)
        derivatives = (tuple(cache.get("T_prime", p, role="training") for p in points)
                       if condition["method"] == "M2" else None)
        training[condition["id"]] = r.SuppliedLabels(points, values, derivatives)
    diag = r.SuppliedLabels(
        diagnostics, tuple(cache.get("T", p, role="diagnostics") for p in diagnostics),
        tuple(cache.get("T_prime", p, role="diagnostics") for p in diagnostics))
    boundaries = r.SuppliedLabels(
        (n.A, n.B), tuple(cache.get("T", p, role="boundaries") for p in (n.A, n.B)),
        tuple(cache.get("T_prime", p, role="boundaries") for p in (n.A, n.B)))
    return training, diag, boundaries


def make_job_context(job, labels, cache, critical_sources):
    adapter = r.LossAdapter(job["method"], labels, job["scales"]["S0"],
                            job["scales"]["S1"], job["scales"]["origin"],
                            n.AffineNormalization(**job["normalization"]))
    context = r.context_from_labels(job, adapter, critical_sources=critical_sources)
    context["label_provenance_signature"] = cache.fingerprint(job["method"], labels.points)
    context["dataset_scope"] = cache.scope
    return adapter, r.validate_context(context)


def select_model(model, tasks, grid):
    """This entire function has no physical-cache argument or oracle access."""
    predictions = sd.cache_predictions(model.forward, grid, source="saved_snapshot")
    selections = [sd.select_grid_minimum(predictions, task["lambda_value"]) for task in tasks]
    estimate = sd.estimate_network_grid_error(model, grid)
    return selections, estimate


def measurement_signature(state, diag, boundaries, tasks, references, cache):
    return r.identity({
        "parameters": state.parameters.to_vector(),
        "diagnostic_labels": diag.data_identity("M2"),
        "boundary_labels": boundaries.data_identity("M2"),
        "diagnostic_label_provenance": cache.fingerprint("M2", diag.points),
        "boundary_label_provenance": cache.fingerprint("M2", boundaries.points),
        "tasks": tasks, "references": references,
    })


def measure_snapshot(state, context, diag, boundaries, tasks, references, cache,
                     *, grid_intervals=1024):
    model = n.NeuralSurrogate(state.parameters, n.AffineNormalization(**context["normalization"]))
    errors_value, errors_derivative = [], []
    for p, target, derivative in zip(diag.points, diag.values, diag.derivatives):
        result = model.evaluate(p)
        errors_value.append(n._finite_result(result.value-target, "diagnostic error"))
        errors_derivative.append(n._finite_result(result.input_derivative-derivative, "diagnostic error"))
    sqrt_n = math.sqrt(len(diag.points))
    rmse0 = n._finite_result(math.hypot(*errors_value)/sqrt_n, "RMSE value")
    rmse1 = n._finite_result(math.hypot(*errors_derivative)/sqrt_n, "RMSE derivative")
    boundary_records = []
    for p, target, derivative in zip(boundaries.points, boundaries.values, boundaries.derivatives):
        result = model.evaluate(p)
        boundary_records.append(dict(p=p, value_error=result.value-target,
                                     derivative_error=result.input_derivative-derivative))
    saturation = saturation_diagnostic(model, diag.points)
    grid = sd.make_grid(n.A, n.B, grid_intervals)
    selections, bound = select_model(model, tasks, grid)
    # All seven choices are already fixed, before any physical decision request.
    decisions = []
    for task, selected in zip(tasks, selections):
        ref = references[task["id"]]
        p = selected.p_hat
        value = cache.get("T", p, role="decisions")
        derivative = cache.get("T_prime", p, role="decisions")
        physical_gradient = n._finite_result(derivative+task["lambda_value"], "physical gradient")
        neural_gradient = n._finite_result(model.input_derivative(p)+task["lambda_value"], "neural gradient")
        objective = n._finite_result(value+task["lambda_value"]*p, "physical objective")
        D = n._finite_result(objective-ref["j_ref"], "raw D")
        expected_branch = ref["branch"]
        branch_mismatch = ((expected_branch == "lower_boundary" and p != n.A)
                           or (expected_branch == "upper_boundary" and p != n.B)
                           or (expected_branch == "interior" and p in (n.A, n.B)))
        decisions.append(dict(
            task=task["id"], task_class=task["task_class"],
            lambda_value=task["lambda_value"], status="AVAILABLE", missing_reason=None,
            p_hat=p, p_ref=ref["p_ref"], physical_branch=expected_branch,
            candidate_origin=selected.candidate_origin,
            predicted_objective=selected.predicted_objective,
            physical_objective=objective, reference_objective=ref["j_ref"], D=D,
            physical_gradient=physical_gradient, neural_gradient=neural_gradient,
            physical_kkt=sd.kkt_residual_interval(p, physical_gradient, n.A, n.B),
            neural_kkt=sd.kkt_residual_interval(p, neural_gradient, n.A, n.B),
            alpha_grid_float=bound["alpha_grid_float"],
            beta_ref_conditional=ref["reference_value_error_estimate_conditional"],
            branch_mismatch=branch_mismatch, D_negative=(D < 0),
            numerical_status="FLOATING_D_NOT_CERTIFIED_REGRET",
            E_D=None, e_hat=None, error_formula="R in [D-E_D,D+E_D+beta_ref] under justified bounds",
            decision_label_provenance=cache.fingerprint("M2", (p,)),
        ))
    primary = [d["D"] for d in decisions if d["task_class"] == "primary_interior"]
    if len(primary) != 3:
        raise IntegrityError("primary score needs all three strictly interior tasks")
    metrics = dict(
        status="AVAILABLE", missing_reason=None,
        parameter_id=r.identity(state.parameters.to_vector()),
        rmse_value=rmse0, rmse_derivative=rmse1,
        rmse_value_normalized=n._finite_result(rmse0/context["scales"]["S0"], "normalised RMSE"),
        rmse_derivative_normalized=n._finite_result(rmse1/context["scales"]["S1"], "normalised RMSE"),
        max_abs_value_error_discrete=max(map(abs, errors_value)),
        max_abs_derivative_error_discrete=max(map(abs, errors_derivative)),
        mean_D_interior=math.fsum(primary)/3.0,
        rounded_saturation_fraction=saturation["rounded_fraction"],
        near_saturation_fraction=saturation["near_fraction"],
        boundary_abs_value_error_a=abs(boundary_records[0]["value_error"]),
        boundary_abs_value_error_b=abs(boundary_records[1]["value_error"]),
        boundary_abs_derivative_error_a=abs(boundary_records[0]["derivative_error"]),
        boundary_abs_derivative_error_b=abs(boundary_records[1]["derivative_error"]),
    )
    counts = dict(
        diagnostic_joint_network_evaluations=len(diag.points),
        boundary_joint_network_evaluations=len(boundaries.points),
        grid_forward_network_evaluations=len(grid.points),
        discrete_objective_evaluations=len(grid.points)*len(tasks),
        decision_input_derivative_evaluations=len(tasks),
        saturation_activation_only_sites=len(diag.points),
        physical_requests_value=len(tasks), physical_requests_derivative=len(tasks))
    return dict(signature=measurement_signature(state, diag, boundaries, tasks, references, cache),
                metrics=metrics, decisions=decisions, boundaries=boundary_records,
                saturation=saturation, grid_error_estimate=bound, counters=counts,
                grid_predictions_reused_across_tasks=True,
                all_selections_completed_before_physical_decision_requests=True)


def validate_measure(measure, state, diag, boundaries, tasks, references, cache):
    expected = measurement_signature(state, diag, boundaries, tasks, references, cache)
    if measure["signature"] != expected:
        raise IntegrityError("saved measurement has incompatible snapshot/data/tasks")
    for decision in measure["decisions"]:
        if decision["decision_label_provenance"] != cache.fingerprint("M2", (decision["p_hat"],)):
            raise IntegrityError("saved decision physical label changed")


def run_job(job, labels, cache, diag, boundaries, tasks, references, critical_sources,
            directory, *, resume=False, save_every=SAVE_EVERY,
            stop_after=None, progress=None, grid_intervals=1024, cache_path=None):
    """Full short integration in tests, or the unchanged 300/3000 recipe in reality.

    stop_after exists for explicit TEST_FIXTURE resource interruption controls,
    never for real performance-based stopping.
    """
    directory = Path(directory)
    if stop_after is not None and cache.scope != "TEST_FIXTURE":
        raise ValueError("only synthetic fixtures may request a short stop")
    adapter, context = make_job_context(job, labels, cache, critical_sources)
    signature = r.identity(context)
    state_path, status_path = directory/"training_state.json", directory/"job_status.json"
    snapshot_name = lambda k: directory/f"snapshot_{k:04d}.json"
    if directory.exists() and not resume:
        raise IntegrityError("new job refuses an existing destination")
    directory.mkdir(parents=True, exist_ok=resume)
    if resume:
        if not state_path.exists() or not status_path.exists():
            raise IntegrityError("interrupted initialization lacks a complete state/status pair")
        cp = r.load_checkpoint(state_path, expected_context=context)
        status = load_record(status_path, signature)
        if status["active_window"] is not None:
            status["uncertain_execution_windows"].append(status["active_window"])
            status["active_window"] = None
        if cp.state.updates < status["persisted_updates"]:
            raise IntegrityError("status ahead of persisted state")
        for k in context["snapshot_updates"]:
            path = snapshot_name(k)
            if path.exists():
                saved = r.load_checkpoint(path, expected_context=context)
                if saved.state != cp.snapshots.get(str(k)):
                    raise IntegrityError("immutable snapshot disagrees with trajectory")
        for k, measure in status["measures"].items():
            validate_measure(measure, cp.snapshots[k], diag, boundaries, tasks, references, cache)
        if status["status"] in ("COMPLETED", "FAILED"):
            if cp.state.updates != status["persisted_updates"]:
                raise IntegrityError("terminal job has mismatched state/status")
            return status
    else:
        initial = r.initialize_parameters(job["seed"])
        cp = r.TrainingCheckpoint.start(
            r.AdamState.zero(initial, r.AdamConfig(**job["adam"])), context)
        status = dict(
            schema_version=SCHEMA, kind="job_status", context_id=signature,
            seed=job["seed"], condition=job["condition"], method=job["method"],
            status="RUNNING", missing_reason=None, persisted_updates=0,
            state_sha256=None, measures={}, active_window=None,
            uncertain_execution_windows=[], updates_executed_this_session=0,
            timings={k: 0.0 for k in ("training", "logging", "metrics", "persistence")},
            timing_scope={"persistence": "training_state writes only; excludes status, cache and snapshot writes"},
            logging_saturation={}, logging_activation_only_sites=0,
            n_value_labels=len(labels.points),
            n_derivative_labels=(len(labels.points) if job["method"] == "M2" else 0),
            C_label_q1_1=(len(labels.points)*(2 if job["method"] == "M2" else 1)))

    def persist_status():
        atomic_record(status_path, status)

    def persist_state():
        started = time.perf_counter()
        r.save_checkpoint(state_path, cp, expected_context=context)
        status["state_sha256"] = file_hash(state_path)
        status["persisted_updates"] = cp.state.updates
        status["counts"] = asdict(cp.counts)
        status["logical_updates"] = cp.state.updates
        status["timings"]["persistence"] += time.perf_counter()-started

    def logging():
        nonlocal cp
        if any(log["updates"] == cp.state.updates for log in cp.logs):
            return
        status["active_window"] = dict(kind="logging", at_step=cp.state.updates,
                                      loss_gradient_calls_at_risk_max=1)
        persist_status()
        started = time.perf_counter()
        cp = r.log_loss_gradient(cp, adapter, expected_context=context)
        if cp.status != "STOPPED_ERROR":
            status["logging_saturation"][str(cp.state.updates)] = saturation_diagnostic(
                n.NeuralSurrogate(cp.state.parameters, adapter.normalization), labels.points)
            status["logging_activation_only_sites"] += len(labels.points)
        status["timings"]["logging"] += time.perf_counter()-started
        status["active_window"] = None

    def snapshot_and_measure():
        k = str(cp.state.updates)
        if cp.state.updates not in context["snapshot_updates"]:
            return
        path = snapshot_name(cp.state.updates)
        if path.exists():
            existing = r.load_checkpoint(path, expected_context=context)
            if existing.state != cp.state:
                raise IntegrityError("snapshot collision")
        else:
            r.save_checkpoint(path, cp, expected_context=context)
        if k in status["measures"]:
            validate_measure(status["measures"][k], cp.state, diag, boundaries, tasks,
                             references, cache)
            return
        status["active_window"] = dict(kind="metrics", at_step=cp.state.updates,
            max_network_calls_at_risk=len(diag.points)+2+grid_intervals+1+len(tasks))
        persist_status()
        started = time.perf_counter()
        status["measures"][k] = measure_snapshot(
            cp.state, context, diag, boundaries, tasks, references, cache,
            grid_intervals=grid_intervals)
        # Persist decision labels before a status can refer to their provenance.
        if cache_path is not None:
            cache.save(cache_path)
        status["timings"]["metrics"] += time.perf_counter()-started
        status["active_window"] = None

    try:
        if cp.state.updates == 0:
            logging()
            persist_state()
            persist_status()
        # A saved milestone can need measurement, but never another update.
        if cp.state.updates in context["snapshot_updates"]:
            snapshot_and_measure()
        while cp.state.updates < context["target_updates"] and cp.status != "STOPPED_ERROR":
            next_snapshot = min(k for k in context["snapshot_updates"] if k > cp.state.updates)
            target = min(cp.state.updates+save_every, next_snapshot,
                         context["target_updates"])
            if stop_after is not None:
                target = min(target, stop_after)
                if target <= cp.state.updates:
                    break
            status["active_window"] = dict(
                kind="training_chunk", from_step=cp.state.updates, to_step=target,
                loss_gradient_calls_at_risk_max=target-cp.state.updates)
            persist_status()
            before = cp.state.updates
            started = time.perf_counter()
            cp = r.continue_to(cp, target, adapter, expected_context=context)
            status["timings"]["training"] += time.perf_counter()-started
            status["updates_executed_this_session"] += cp.state.updates-before
            if cp.status != "STOPPED_ERROR" and cp.state.updates in context["snapshot_updates"]:
                logging()
            persist_state()
            status["active_window"] = None
            if cp.status != "STOPPED_ERROR":
                snapshot_and_measure()
            persist_status()
            if progress and cp.state.updates in context["snapshot_updates"]:
                progress(job, cp.state.updates, status)
            if stop_after is not None and cp.state.updates >= stop_after:
                break
        if cp.status == "STOPPED_ERROR":
            status["status"], status["missing_reason"] = "FAILED", cp.incident
        elif cp.state.updates == context["target_updates"]:
            status["status"] = "COMPLETED"
        else:
            status["status"] = "PAUSED_FIXTURE"
    except IntegrityError:
        raise
    except (ArithmeticError, ValueError) as exc:
        # State/context failures are not silently reclassified as bad scores.
        if isinstance(exc, ValueError):
            raise IntegrityError(str(exc)) from exc
        status["status"] = "FAILED"
        status["missing_reason"] = dict(stage="measurement_or_numeric",
                                        error_type=type(exc).__name__, message=str(exc))
        # A measurement failure preserves the last valid training checkpoint.
        # In particular a completed checkpoint must never be rewritten as failed.
        # The job status carries the failure and explicitly missing measurements.
    status["active_window"] = None
    status["snapshot_steps_available"] = sorted(map(int, cp.snapshots))
    status["missing_snapshot_steps"] = [k for k in context["snapshot_updates"] if str(k) not in cp.snapshots]
    status["logs"] = list(cp.logs)
    persist_status()
    return status


def aggregate_rows(jobs, statuses, tasks):
    """Reserve every planned key, including failed/missing measurements."""
    metric_rows, decision_rows = [], []
    for job in jobs:
        key = (job["seed"], job["condition"])
        status = statuses.get(key)
        for step in job["snapshot_updates"]:
            measure = status["measures"].get(str(step)) if status else None
            base = dict(seed=key[0], condition=key[1], step=step)
            if measure:
                metric = dict(base, **measure["metrics"])
                metric.update(training_loss=next(
                    (log["loss"] for log in status["logs"] if log["updates"] == step), None),
                    n_value_labels=status["n_value_labels"],
                    n_derivative_labels=status["n_derivative_labels"],
                    C_label_q1_1=status["C_label_q1_1"])
                for decision in measure["decisions"]:
                    decision_rows.append(dict(base, **decision))
            else:
                reason = ("job not processed" if not status else str(status["missing_reason"]
                           or "snapshot/measurement not reached"))
                metric = dict(base, status="MISSING", missing_reason=reason)
                for task in tasks:
                    decision_rows.append(dict(base, task=task["id"],
                        task_class=task["task_class"], lambda_value=task["lambda_value"],
                        status="MISSING", missing_reason=reason))
            metric_rows.append(metric)
    for rows, fields in ((metric_rows, ("seed","condition","step")),
                         (decision_rows, ("seed","condition","step","task"))):
        if len({tuple(row[k] for k in fields) for row in rows}) != len(rows):
            raise IntegrityError("duplicate aggregate key")
    return metric_rows, decision_rows


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=path.name+".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                values = {}
                for key in fields:
                    value = row.get(key)
                    if isinstance(value, float):
                        n._finite_scalar(value, key)
                        value = repr(value)
                    values[key] = "" if value is None else value
                writer.writerow(values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def descriptive_table(metrics):
    result = []
    for condition in r.CONDITIONS:
        rows = [m for m in metrics if m["condition"] == condition and m["step"] == 3000
                and m["status"] == "AVAILABLE"]
        result.append(dict(condition=condition, valid=len(rows), missing=20-len(rows),
            mean_rmse_value=(math.fsum(m["rmse_value"] for m in rows)/len(rows) if rows else None),
            mean_rmse_derivative=(math.fsum(m["rmse_derivative"] for m in rows)/len(rows) if rows else None),
            mean_D_interior=(math.fsum(m["mean_D_interior"] for m in rows)/len(rows) if rows else None)))
    return result
