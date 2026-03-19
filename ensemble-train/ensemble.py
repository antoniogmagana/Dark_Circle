"""
Sensor Ensemble — weighted late fusion across per-sensor models.

Architecture:
  Stage 1 (Detection):  One model per sensor votes on [background, vehicle].
  Stage 2 (Classification): One model per sensor votes on vehicle category/instance.
  Fusion: Weighted average of softmax scores.  Weights default to each model's
          validation F1, normalised to sum to 1.  Missing sensors are handled
          gracefully by renormalising over whatever sensors are available.

Usage:
  # After training + evaluation:
  python ensemble.py build          # discover best models, compute weights, save
  python ensemble.py show           # print the current ensemble configuration
  python ensemble.py eval           # run two-stage ensemble evaluation on test set

  # Programmatic (for live inference):
  from ensemble import SensorEnsemble
  ens = SensorEnsemble.load()
  result = ens.two_stage_predict({"seismic": tensor, "audio": tensor})
"""

import json
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from models import build_model
from preprocess import preprocess


# =====================================================================
# Constants
# =====================================================================

ALL_SENSORS = ["audio", "seismic", "accel"]
ALL_MODES = ["detection", "category", "instance"]

DEFAULT_MODEL_DIR = Path("saved_models")
WEIGHTS_FILENAME = "ensemble_weights.json"


# =====================================================================
# Model Discovery
# =====================================================================

def parse_eval_report(report_path):
    """Extract F1-Score from an evaluation_report.txt file."""
    try:
        text = report_path.read_text()
        match = re.search(r"F1-Score:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0


def discover_best_models(model_dir=DEFAULT_MODEL_DIR):
    """
    Scan saved_models/{mode}/{sensor}/{model_name}/{run_id}/ and return
    the best run (by F1) for each (mode, sensor) pair.

    Returns:
        dict: {mode: {sensor: {"run_dir": Path, "f1": float, "model_name": str}}}
    """
    best = {}

    for mode in ALL_MODES:
        best[mode] = {}
        for sensor in ALL_SENSORS:
            sensor_dir = model_dir / mode / sensor
            if not sensor_dir.exists():
                continue

            best_f1 = -1.0
            best_entry = None

            # Walk: sensor_dir / model_name / run_id /
            for report_path in sensor_dir.rglob("evaluation_report.txt"):
                run_dir = report_path.parent
                model_path = run_dir / "best_model.pth"
                json_path = run_dir / "hyperparameters.json"

                if not model_path.exists() or not json_path.exists():
                    continue

                f1 = parse_eval_report(report_path)
                if f1 > best_f1:
                    best_f1 = f1

                    # Extract model name from the directory structure
                    # Path: .../{sensor}/{model_name}/{run_id}/
                    model_name = run_dir.parent.name

                    best_entry = {
                        "run_dir": str(run_dir),
                        "f1": f1,
                        "model_name": model_name,
                    }

            if best_entry is not None:
                best[mode][sensor] = best_entry

    return best


# =====================================================================
# Ensemble Class
# =====================================================================

class SensorEnsemble:
    """
    Weighted late-fusion ensemble across per-sensor models.

    Attributes:
        models:  {mode: {sensor: (nn.Module, SimpleNamespace)}}
        weights: {mode: {sensor: float}}  — normalised per mode
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.models = {}        # loaded pytorch models + their configs
        self.weights = {}       # fusion weights per (mode, sensor)
        self.model_info = {}    # metadata from discovery

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    @classmethod
    def build(cls, model_dir=DEFAULT_MODEL_DIR, device=None):
        """Discover best models, load them, compute weights from F1."""
        ens = cls(device=device)
        ens.model_info = discover_best_models(model_dir)

        for mode in ALL_MODES:
            ens.models[mode] = {}
            ens.weights[mode] = {}

            mode_info = ens.model_info.get(mode, {})
            if not mode_info:
                continue

            # Load each sensor's best model
            for sensor, info in mode_info.items():
                run_dir = Path(info["run_dir"])
                try:
                    model, cfg = _load_model_from_run(run_dir, ens.device)
                    ens.models[mode][sensor] = (model, cfg)
                except Exception as e:
                    print(f"  [!] Failed to load {mode}/{sensor}: {e}")
                    continue

            # Compute weights from F1 scores (softmax normalisation)
            _compute_f1_weights(ens, mode, mode_info)

        return ens

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save_weights(self, path=None):
        """Save fusion weights and model info to JSON."""
        path = path or DEFAULT_MODEL_DIR / WEIGHTS_FILENAME
        data = {
            "weights": self.weights,
            "model_info": self.model_info,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved ensemble config to: {path}")

    @classmethod
    def load(cls, path=None, device=None):
        """Load a previously saved ensemble (weights + models)."""
        path = path or DEFAULT_MODEL_DIR / WEIGHTS_FILENAME
        with open(path) as f:
            data = json.load(f)

        ens = cls(device=device)
        ens.weights = data["weights"]
        ens.model_info = data["model_info"]

        # Reload all referenced models
        for mode, sensors in ens.model_info.items():
            ens.models[mode] = {}
            for sensor, info in sensors.items():
                run_dir = Path(info["run_dir"])
                try:
                    model, cfg = _load_model_from_run(run_dir, ens.device)
                    ens.models[mode][sensor] = (model, cfg)
                except Exception as e:
                    print(f"  [!] Failed to load {mode}/{sensor}: {e}")

        return ens

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(self, sensor_data, mode):
        """
        Run weighted ensemble for a single mode.

        Args:
            sensor_data: dict of {sensor_name: tensor [C, T]}
                         Each tensor is a raw 1-second window at native sample rate.
            mode: "detection", "category", or "instance"

        Returns:
            fused_probs: tensor [NUM_CLASSES] — weighted average of softmax scores
        """
        if mode not in self.models or not self.models[mode]:
            raise ValueError(f"No models loaded for mode '{mode}'")

        scores = {}
        for sensor, tensor in sensor_data.items():
            if sensor not in self.models[mode]:
                continue

            model, cfg = self.models[mode][sensor]
            x = tensor.unsqueeze(0).to(self.device)     # [1, C, T]
            x = preprocess(x, config=cfg)

            with torch.inference_mode():
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            scores[sensor] = probs.squeeze(0)            # [NUM_CLASSES]

        if not scores:
            raise ValueError(
                f"No models available for sensors: {list(sensor_data.keys())} "
                f"in mode '{mode}'"
            )

        # Renormalise weights over available sensors
        mode_weights = self.weights.get(mode, {})
        available = [s for s in scores if s in mode_weights]

        if not available:
            # Fallback: equal weight for all available sensors
            available = list(scores.keys())
            w = torch.ones(len(available), device=self.device) / len(available)
        else:
            w = torch.tensor(
                [mode_weights[s] for s in available], device=self.device
            )
            w = w / w.sum()

        fused = sum(w[i] * scores[s] for i, s in enumerate(available))
        return fused

    def two_stage_predict(self, sensor_data, detection_threshold=0.5):
        """
        Two-stage pipeline: detect vehicle → classify type.

        Args:
            sensor_data: dict of {sensor_name: tensor [C, T]}
            detection_threshold: confidence threshold for stage 1

        Returns:
            dict with keys: detected, detection_confidence,
                            and (if detected) class_id, class_confidence, class_probs
        """
        # Stage 1: Detection
        det_probs = self.predict(sensor_data, mode="detection")
        vehicle_conf = det_probs[1].item()   # class 1 = vehicle

        if vehicle_conf < detection_threshold:
            return {
                "detected": False,
                "detection_confidence": 1.0 - vehicle_conf,
            }

        result = {
            "detected": True,
            "detection_confidence": vehicle_conf,
        }

        # Stage 2: Classification (try category, then instance)
        for mode in ("category", "instance"):
            if mode in self.models and self.models[mode]:
                try:
                    cls_probs = self.predict(sensor_data, mode=mode)
                    cls_id = cls_probs.argmax().item()
                    result[f"{mode}_class_id"] = cls_id
                    result[f"{mode}_confidence"] = cls_probs[cls_id].item()
                    result[f"{mode}_probs"] = cls_probs.cpu().numpy()
                except ValueError:
                    pass

        return result

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------

    def summary(self):
        """Print the current ensemble configuration."""
        sep = "=" * 70
        print(f"\n{sep}")
        print(f"{'SENSOR ENSEMBLE CONFIGURATION':^70}")
        print(sep)

        for mode in ALL_MODES:
            sensors = self.weights.get(mode, {})
            if not sensors:
                print(f"\n  {mode.upper()}: (no models)")
                continue

            print(f"\n  {mode.upper()}:")
            print(f"    {'Sensor':<12} {'Model':<30} {'F1':<8} {'Weight':<8}")
            print(f"    {'-'*58}")

            info = self.model_info.get(mode, {})
            for sensor in ALL_SENSORS:
                if sensor not in sensors:
                    continue
                w = sensors[sensor]
                f1 = info.get(sensor, {}).get("f1", 0.0)
                name = info.get(sensor, {}).get("model_name", "?")
                print(f"    {sensor:<12} {name:<30} {f1:<8.4f} {w:<8.4f}")

        print(f"\n{sep}\n")


# =====================================================================
# Internal Helpers
# =====================================================================

def _load_model_from_run(run_dir, device):
    """Load a trained model and its config from a run directory."""
    json_path = run_dir / "hyperparameters.json"
    with open(json_path) as f:
        config_dict = json.load(f)

    if "CLASS_MAP" in config_dict:
        config_dict["CLASS_MAP"] = {int(k): v for k, v in config_dict["CLASS_MAP"].items()}

    cfg = SimpleNamespace(**config_dict)
    cfg.DEVICE = str(device)

    # Recover USE_MEL from meta.pt if available
    meta_path = run_dir / "meta.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, map_location=device, weights_only=False)
        cfg.USE_MEL = meta.get("use_mel", getattr(cfg, "USE_MEL", True))

    model = build_model(
        input_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        config=cfg,
    ).to(device)

    model_path = run_dir / "best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, cfg


def _compute_f1_weights(ens, mode, mode_info):
    """
    Set per-sensor weights proportional to softmax of F1 scores.

    Using softmax rather than raw normalisation prevents a single sensor
    with F1=0 from collapsing the weights and provides a smoother
    distribution when scores are close.
    """
    available = [s for s in mode_info if s in ens.models.get(mode, {})]
    if not available:
        return

    f1_scores = np.array([mode_info[s]["f1"] for s in available])

    # Softmax: exp(f1) / sum(exp(f1))  — temperature 1.0
    exp_scores = np.exp(f1_scores - f1_scores.max())  # numerical stability
    weights = exp_scores / exp_scores.sum()

    ens.weights[mode] = {s: float(w) for s, w in zip(available, weights)}


# =====================================================================
# CLI
# =====================================================================

def cmd_build():
    """Discover best models and compute ensemble weights."""
    print("Discovering best models per (mode, sensor)...")
    ens = SensorEnsemble.build()
    ens.save_weights()
    ens.summary()


def cmd_show():
    """Display the current ensemble configuration."""
    path = DEFAULT_MODEL_DIR / WEIGHTS_FILENAME
    if not path.exists():
        print(f"No ensemble config found at {path}. Run 'python ensemble.py build' first.")
        return
    ens = SensorEnsemble.load()
    ens.summary()


def cmd_eval():
    """
    Evaluate the ensemble on the test set.

    This runs the full two-stage pipeline: for each test sample, all
    available sensor models vote, and the fused predictions are scored.
    
    NOTE: This requires aligned multi-sensor test data. For now it
    evaluates each mode independently using per-sensor predictions.
    Full multi-sensor aligned evaluation can be added as a follow-up.
    """
    path = DEFAULT_MODEL_DIR / WEIGHTS_FILENAME
    if not path.exists():
        print(f"No ensemble config found. Run 'python ensemble.py build' first.")
        return

    ens = SensorEnsemble.load()

    print("\nEnsemble Evaluation (per-mode weighted fusion)")
    print("=" * 70)

    for mode in ALL_MODES:
        mode_info = ens.model_info.get(mode, {})
        mode_weights = ens.weights.get(mode, {})

        if not mode_info or not mode_weights:
            print(f"\n  {mode.upper()}: (no models — skipping)")
            continue

        # Load predictions from each sensor model
        sensor_preds = {}
        common_labels = None

        for sensor, info in mode_info.items():
            pred_path = Path(info["run_dir"]) / "predictions.npz"
            if not pred_path.exists():
                print(f"  [!] Missing predictions.npz for {mode}/{sensor}")
                continue

            data = np.load(pred_path)
            labels = data["labels"]
            probs = data["probs"]

            # For fusion to work, all sensors need the same sample count.
            # If counts differ (different test splits), we use the minimum
            # intersection length as a rough alignment.
            if common_labels is None:
                common_labels = labels
            elif len(labels) != len(common_labels):
                min_len = min(len(labels), len(common_labels))
                labels = labels[:min_len]
                probs = probs[:min_len]
                common_labels = common_labels[:min_len]
                print(f"    [!] Sample count mismatch for {sensor} — truncating to {min_len}")

            sensor_preds[sensor] = probs

        if not sensor_preds or common_labels is None:
            print(f"\n  {mode.upper()}: (no predictions available)")
            continue

        # Weighted fusion
        available = [s for s in sensor_preds if s in mode_weights]
        w = np.array([mode_weights[s] for s in available])
        w = w / w.sum()

        fused = sum(w[i] * sensor_preds[s] for i, s in enumerate(available))
        fused_preds = fused.argmax(axis=1)

        from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

        acc = accuracy_score(common_labels, fused_preds)
        f1 = f1_score(common_labels, fused_preds, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(common_labels, fused_preds)

        print(f"\n  {mode.upper()} ENSEMBLE:")
        print(f"    Sensors:  {', '.join(available)}")
        print(f"    Weights:  {', '.join(f'{s}={w[i]:.3f}' for i, s in enumerate(available))}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1:       {f1:.4f}")
        print(f"    MCC:      {mcc:.4f}")

        # Compare against individual models
        print(f"\n    Individual model scores:")
        for sensor in available:
            individual_preds = sensor_preds[sensor].argmax(axis=1)
            ind_f1 = f1_score(common_labels, individual_preds, average="weighted", zero_division=0)
            print(f"      {sensor:<12} F1: {ind_f1:.4f}")

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ensemble.py [build|show|eval]")
        return

    cmd = sys.argv[1].lower()
    if cmd == "build":
        cmd_build()
    elif cmd == "show":
        cmd_show()
    elif cmd == "eval":
        cmd_eval()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python ensemble.py [build|show|eval]")


if __name__ == "__main__":
    main()
