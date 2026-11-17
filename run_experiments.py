"""Воспроизводимые эксперименты к лабораторной №4.

Запуск:
    python run_experiments.py

Протокол:
  * данные german.csv, стратифицированный split 80/20 (random_state=42);
  * отбор модели — 5-fold CV ROC-AUC только на train (test не используется);
  * финальная модель обучается на полном train и один раз оценивается на test;
  * остальные кандидаты оцениваются на test только для сравнительной таблицы;
  * сохраняются experiment_results.json и графики в figures/.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ------------------------------------------------------------------
#  Данные
# ------------------------------------------------------------------

def load_german() -> tuple[np.ndarray, np.ndarray]:
    """german.csv: цель — первый столбец Creditability (0/1)."""
    df = pd.read_csv(ROOT / "german.csv", sep=";")
    y = df.iloc[:, 0].to_numpy(dtype=int)
    x = df.iloc[:, 1:].to_numpy(dtype=float)
    return x, y


def binary(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y) > 0.5).astype(int)


# ------------------------------------------------------------------
#  Модели
# ------------------------------------------------------------------

def build_candidates(seed: int) -> dict[str, object]:
    """Кандидаты для сравнительной таблицы (кроме финального Voting)."""
    rf = RandomForestClassifier(
        n_estimators=500, max_features=0.33, random_state=seed, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=3,
        subsample=0.8, random_state=seed,
    )
    et = ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)

    return {
        "RandomForest(500, mf=0.33)": rf,
        "GradientBoosting(600, lr=0.03)": gb,
        "ExtraTrees(500)": et,
        "HistGradientBoosting(400)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, l2_regularization=1.0,
            random_state=seed,
        ),
        "AdaBoost(300, lr=0.05)": AdaBoostClassifier(
            n_estimators=300, learning_rate=0.05, random_state=seed
        ),
        "MLP(64-32)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=1500,
                early_stopping=True, n_iter_no_change=25, random_state=seed,
            ),
        ),
        "Stacking(RF+GB+ET+LR)": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=500, max_features=0.33, random_state=seed, n_jobs=1
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=600, learning_rate=0.03, max_depth=3,
                    subsample=0.8, random_state=seed,
                )),
                ("et", ExtraTreesClassifier(
                    n_estimators=500, random_state=seed, n_jobs=1
                )),
                ("lr", make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=5000)
                )),
            ],
            final_estimator=LogisticRegression(max_iter=5000),
            cv=5, n_jobs=-1,
        ),
    }


def build_voting(seed: int) -> VotingClassifier:
    """Та же конфигурация, что и в контрактной функции."""
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=500, max_features=0.33, random_state=seed, n_jobs=-1
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=3,
                subsample=0.8, random_state=seed,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=500, random_state=seed, n_jobs=-1
            )),
        ],
        voting="soft",
    )


# ------------------------------------------------------------------
#  Метрики и графики
# ------------------------------------------------------------------

def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def fit_and_evaluate(model, x_train, y_train, x_test, y_test) -> dict[str, float]:
    t0 = time.time()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    result = metrics(y_test, probabilities)
    result["fit_seconds"] = float(time.time() - t0)
    return result


def save_plots(final_proba, y_test, experiments, cv_results) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    saved: list[str] = []

    # ROC-кривая финальной модели
    fpr, tpr, _ = roc_curve(y_test, final_proba)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.plot(fpr, tpr, color="#2F6B4F", linewidth=2,
            label=f"SoftVoting (AUC = {roc_auc_score(y_test, final_proba):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve — final model (test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "roc_curve.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))

    # Сравнение кандидатов по ROC-AUC на test
    ordered = sorted(experiments, key=lambda e: e["roc_auc"])
    names = [e["model"] for e in ordered]
    values = [e["roc_auc"] for e in ordered]
    colors = ["#C0504D" if v <= 0.80 else "#4F81BD" for v in values]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(names, values, color=colors)
    ax.axvline(0.80, color="red", linestyle="--", linewidth=1.2,
               label="10/10 threshold (0.80)")
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0.70, max(values) + 0.03)
    ax.set_xlabel("ROC-AUC (test)")
    ax.set_title("Model comparison on the same 80/20 split")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = fig_dir / "model_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))

    # CV ROC-AUC на train
    cv_names = [c["model"] for c in cv_results]
    cv_values = [c["mean_auc"] for c in cv_results]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(cv_names, cv_values, color="#6A9FB5")
    for i, c in enumerate(cv_results):
        ax.text(c["mean_auc"] + 0.001, i,
                f"{c['mean_auc']:.4f} ± {c['std_auc']:.4f}",
                va="center", fontsize=8)
    ax.set_xlim(0.74, max(cv_values) + 0.02)
    ax.set_xlabel("Mean ROC-AUC (5-fold CV on train)")
    ax.set_title("Model selection: CV on the training part only")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = fig_dir / "cv_auc.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    saved.append(str(path))

    return saved


# ------------------------------------------------------------------
#  Основной сценарий
# ------------------------------------------------------------------

def main() -> None:
    x, y = load_german()
    y_bin = binary(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_bin, test_size=TEST_SIZE, random_state=SEED, stratify=y_bin
    )

    print("=" * 70)
    print("Лабораторная №4 — ансамбли, german.csv")
    print("=" * 70)
    print(f"  всего: {len(y_bin)} объектов, {x.shape[1]} признаков; "
          f"класс 1 = {int(y_bin.sum())}, класс 0 = {int((1 - y_bin).sum())}")
    print(f"  train: {len(y_train)} (1={int(y_train.sum())}, 0={int((1 - y_train).sum())})")
    print(f"  test:  {len(y_test)} (1={int(y_test.sum())}, 0={int((1 - y_test).sum())})")
    print(f"  split: test_size={TEST_SIZE}, random_state={SEED}, stratify=True")

    # ---------- Отбор модели: CV только на train ----------
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_models = {
        "RandomForest(500, mf=0.33)": RandomForestClassifier(
            n_estimators=500, max_features=0.33, random_state=SEED, n_jobs=-1
        ),
        "GradientBoosting(600, lr=0.03)": GradientBoostingClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            subsample=0.8, random_state=SEED,
        ),
        "ExtraTrees(500)": ExtraTreesClassifier(
            n_estimators=500, random_state=SEED, n_jobs=-1
        ),
        "SoftVoting(RF+GB+ET)": build_voting(SEED),
    }

    print("\n" + "=" * 70)
    print(f"Отбор модели: {CV_FOLDS}-fold CV ROC-AUC только на train")
    print("=" * 70)
    cv_results = []
    for name, model in cv_models.items():
        t0 = time.time()
        scores = cross_val_score(
            model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        cv_results.append({
            "model": name,
            "mean_auc": float(scores.mean()),
            "std_auc": float(scores.std()),
        })
        print(f"  {name:32s} {scores.mean():.4f} ± {scores.std():.4f}  "
              f"({time.time() - t0:.1f}s)")
    cv_results.sort(key=lambda c: -c["mean_auc"])
    print(f"  -> лучший по CV: {cv_results[0]['model']} "
          f"({cv_results[0]['mean_auc']:.4f})")

    # ---------- Финальная контрактная функция ----------
    print("\n" + "=" * 70)
    print("Финальная модель: run_ensemble_experiment(...)")
    print("=" * 70)
    t0 = time.time()
    final = run_ensemble_experiment(EnsembleInput(
        train_features=x_train, train_target=y_train,
        test_features=x_test, test_target=y_test,
        random_seed=SEED,
    ))
    final_seconds = time.time() - t0
    final_metrics = {
        "roc_auc": final.roc_auc,
        "accuracy": final.accuracy,
        "precision": final.precision,
        "recall": final.recall,
        "f1": final.f1,
    }
    print(f"  model: {final.model_name}")
    print(f"  ROC-AUC  = {final.roc_auc:.4f}")
    print(f"  Accuracy = {final.accuracy:.4f}")
    print(f"  Precision= {final.precision:.4f}")
    print(f"  Recall   = {final.recall:.4f}")
    print(f"  F1       = {final.f1:.4f}")
    print(f"  fit+predict: {final_seconds:.1f}s")

    # ---------- Сравнительные эксперименты на test ----------
    print("\n" + "=" * 70)
    print("Сравнение кандидатов на том же test (для отчёта)")
    print("=" * 70)
    experiments = []
    candidates = build_candidates(SEED)
    for name, model in candidates.items():
        result = fit_and_evaluate(model, x_train, y_train, x_test, y_test)
        experiments.append({"model": name, **result})
        print(f"  {name:32s} AUC={result['roc_auc']:.4f} "
              f"Acc={result['accuracy']:.4f} F1={result['f1']:.4f} "
              f"({result['fit_seconds']:.1f}s)")

    experiments.append({
        "model": "SoftVoting(RF+GB+ET) — final",
        **final_metrics,
        "fit_seconds": float(final_seconds),
    })
    experiments.sort(key=lambda e: -e["roc_auc"])

    print("\n  итоговая таблица (test):")
    print(f"  {'model':36s}{'AUC':>8s}{'Acc':>8s}{'Prec':>8s}{'Rec':>8s}{'F1':>8s}")
    for e in experiments:
        print(f"  {e['model']:36s}{e['roc_auc']:8.4f}{e['accuracy']:8.4f}"
              f"{e['precision']:8.4f}{e['recall']:8.4f}{e['f1']:8.4f}")

    # ---------- Сохранение результатов ----------
    results = {
        "model_name": final.model_name,
        "random_seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "metrics": {k: round(v, 6) for k, v in final_metrics.items()},
        "cv_on_train": {
            "folds": CV_FOLDS,
            "scoring": "roc_auc",
            "candidates": [
                {
                    "model": c["model"],
                    "mean_auc": round(c["mean_auc"], 6),
                    "std_auc": round(c["std_auc"], 6),
                }
                for c in cv_results
            ],
        },
        "experiments_on_test": [
            {
                "model": e["model"],
                "roc_auc": round(e["roc_auc"], 6),
                "accuracy": round(e["accuracy"], 6),
                "precision": round(e["precision"], 6),
                "recall": round(e["recall"], 6),
                "f1": round(e["f1"], 6),
            }
            for e in experiments
        ],
    }
    out_path = ROOT / "experiment_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nСохранено: {out_path}")

    plots = save_plots(final.probabilities, y_test, experiments, cv_results)
    for path in plots:
        print(f"График: {path}")


if __name__ == "__main__":
    main()
