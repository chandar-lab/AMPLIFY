import json
import os
import time
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator

# NVML is optional: power telemetry is skipped when the bindings are unavailable.
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False


class _PowerMeter:
    """Samples board power for the local GPU via NVML and integrates it into energy.

    One handle per process; with one process per GPU, summing across ranks gives the
    node-level draw.  Energy is counted per job -- it restarts at zero on resume.
    """

    def __init__(self) -> None:
        self._handle = None
        self._last_time = time.perf_counter()
        self.energy_joules = 0.0
        if _nvml_available and torch.cuda.is_available():
            try:
                self._handle = self._local_handle()
            except Exception:
                self._handle = None

    @staticmethod
    def _local_handle():
        """Resolve the NVML handle for this process's GPU.

        Under SLURM, ``CUDA_VISIBLE_DEVICES`` renumbers devices, so the torch index is
        not the NVML index.  Match on UUID and fall back to the index only if that fails.
        """
        index = torch.cuda.current_device()
        uuid = str(getattr(torch.cuda.get_device_properties(index), "uuid", ""))
        if uuid:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                nvml_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(nvml_uuid, bytes):
                    nvml_uuid = nvml_uuid.decode()
                if uuid in nvml_uuid:
                    return handle
        return pynvml.nvmlDeviceGetHandleByIndex(index)

    def sample(self) -> Optional[float]:
        """Return instantaneous board power in watts, accumulating energy since the last call."""
        if self._handle is None:
            return None
        now = time.perf_counter()
        watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        self.energy_joules += watts * (now - self._last_time)
        self._last_time = now
        return watts


class TrainingLogger:
    """Checkpointable metric logger for distributed training.

    Train metrics are accumulated on GPU across all micro-batches between logging steps,
    then averaged.  A single all-reduce at log time synchronises counters across ranks, so
    there are no ``.item()`` calls in the training loop.  Eval metrics are accumulated over
    the full eval dataset.

    Logged metrics are deliberately limited to loss, expected recursion depth, eval accuracy,
    throughput and GPU power.  Throughput is what makes the effect
    of FlashAttention, packing and ``torch.compile`` directly observable.
    """

    _STATE_KEYS = ("num_steps", "num_epochs", "num_batches_in_epoch", "num_tokens")
    _TRAIN_KEYS = ("tokens", "pred", "loss_sum", "mlm_loss_sum", "cycles_sum", "cycles_n")

    def __init__(self) -> None:
        self.num_steps = 0
        self.num_epochs = 0
        self.num_batches_in_epoch = 0
        self.num_tokens = 0
        self.steps_per_epoch: float = 1.0
        self._train: Dict[str, torch.Tensor] = {}
        self._eval: Dict[str, Dict[str, float]] = {}
        self._power = _PowerMeter()
        self._last_log_time = time.perf_counter()
        self._last_log_step = 0
        self._reset()

    def _reset(self) -> None:
        self._train = {k: torch.tensor(0.0, dtype=torch.float64) for k in self._TRAIN_KEYS}
        self._eval.clear()

    def _ensure_device(self, device: torch.device) -> None:
        if self._train["pred"].device != device:
            self._train = {k: v.to(device) for k, v in self._train.items()}

    def state_dict(self) -> Dict[str, int]:
        return {k: getattr(self, k) for k in self._STATE_KEYS}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        for k in self._STATE_KEYS:
            setattr(self, k, state.get(k, 0))
        # Restart the throughput window so the first post-resume value is not skewed
        # by the time spent loading the checkpoint.
        self._last_log_time = time.perf_counter()
        self._last_log_step = self.num_steps

    def add_train_batch(self, batch: Dict[str, Any], output: Any, total_loss: torch.Tensor) -> None:
        """Accumulate train metrics on GPU.

        Args:
            batch: Collated batch (provides token counts and labels).
            output: Model output with ``loss`` (MLM CE) and optional ``avg_cycles``.
            total_loss: MLM loss plus the weighted ponder cost -- the quantity optimised.
        """
        labels = batch["labels"]
        self._ensure_device(labels.device)

        num_pred = labels.ne(-100).sum()
        t = self._train
        # Packing mode: the collator provides exact counts.
        t["tokens"] += batch["num_tokens"] if "num_tokens" in batch else batch["attention_mask"].sum()
        t["pred"] += num_pred
        t["loss_sum"] += total_loss.detach() * num_pred
        t["mlm_loss_sum"] += output.loss.detach() * num_pred
        # avg_cycles only exists for the adaptive GMR variants; plain AMPLIFY and the
        # fixed-depth baseline share this logger and do not report it.
        if getattr(output, "avg_cycles", None) is not None:
            t["cycles_sum"] += output.avg_cycles
            t["cycles_n"] += 1

    def add_eval_batch(self, split: str, batch: Dict[str, Any], output: Any) -> None:
        """Accumulate eval metrics over the full eval dataset."""
        labels = batch["labels"]
        mask = labels.ne(-100)
        num_pred = mask.sum()
        if split not in self._eval:
            self._eval[split] = {"pred": 0.0, "loss_sum": 0.0, "correct_sum": 0.0, "cycles_sum": 0.0, "cycles_n": 0.0}
        e = self._eval[split]
        e["pred"] += num_pred.item()
        e["loss_sum"] += (output.loss.detach() * num_pred).item()
        # Masked-token recovery rate: the interpretable companion to eval loss.
        e["correct_sum"] += output.logits.detach().argmax(dim=-1).eq(labels).logical_and(mask).sum().item()
        if getattr(output, "avg_cycles", None) is not None:
            e["cycles_sum"] += output.avg_cycles.item()
            e["cycles_n"] += 1.0

    def log(self, optimizer: torch.optim.Optimizer, accelerator: Accelerator, json_path: str) -> None:
        """Reduce accumulated metrics across ranks, log, and reset."""
        device = accelerator.device

        # Pack train accumulators, power telemetry and eval values into one all-reduce.
        to_reduce: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in self._train.items()}
        watts = self._power.sample()
        to_reduce["power_w"] = torch.tensor(watts or 0.0, dtype=torch.float64, device=device)
        to_reduce["power_n"] = torch.tensor(0.0 if watts is None else 1.0, dtype=torch.float64, device=device)
        to_reduce["energy_j"] = torch.tensor(self._power.energy_joules, dtype=torch.float64, device=device)
        for name in self._eval:
            for k, v in self._eval[name].items():
                to_reduce[f"{name}/{k}"] = torch.tensor(v, dtype=torch.float64, device=device)

        keys = list(to_reduce.keys())
        r = dict(zip(keys, accelerator.reduce(torch.stack(list(to_reduce.values())), reduction="sum").cpu().tolist()))

        self.num_tokens += int(r["tokens"])

        # Throughput over the window since the last log, summed across ranks.
        now = time.perf_counter()
        elapsed = max(now - self._last_log_time, 1e-12)
        steps_done = self.num_steps - self._last_log_step

        metrics: Dict[str, float] = {
            "train/epoch": self.num_steps / self.steps_per_epoch,
            "train/global_step": self.num_steps,
            "train/tokens": self.num_tokens,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/loss": r["loss_sum"] / r["pred"],
            "train/mlm_loss": r["mlm_loss_sum"] / r["pred"],
            "train/tokens_per_second": r["tokens"] / elapsed,
            "train/steps_per_second": steps_done / elapsed,
            "train/ms_per_step": 1e3 * elapsed / max(steps_done, 1),
        }
        if r["cycles_n"] > 0:
            metrics["train/avg_cycles"] = r["cycles_sum"] / r["cycles_n"]
        if r["power_n"] > 0:
            # Sum over ranks (one GPU each) gives node-level draw; energy in kWh.
            metrics["train/gpu_power_w"] = r["power_w"]
            metrics["train/gpu_energy_kwh"] = r["energy_j"] / 3.6e6

        for name in self._eval:
            pred = r[f"{name}/pred"]
            if pred <= 0:
                continue
            metrics[f"eval/{name}/loss"] = r[f"{name}/loss_sum"] / pred
            metrics[f"eval/{name}/accuracy"] = r[f"{name}/correct_sum"] / pred
            if r[f"{name}/cycles_n"] > 0:
                metrics[f"eval/{name}/avg_cycles"] = r[f"{name}/cycles_sum"] / r[f"{name}/cycles_n"]

        accelerator.log(metrics, step=self.num_steps)

        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "a") as f:
                f.write(json.dumps({k: float(v) for k, v in metrics.items()}) + "\n")

        self._last_log_time = now
        self._last_log_step = self.num_steps
        self._reset()
