"""Лабораторная №4 — ансамбль на german.csv.

Финальная модель — мягкое голосование (soft voting) трёх разнородных
ансамблей:

  * Random Forest (500 деревьев, max_features=0.33);
  * Gradient Boosting (600 деревьев, lr=0.03, depth=3, subsample=0.8);
  * Extra Trees (500 деревьев, max_features='sqrt').

Все базовые модели обучаются только на train-части. ``test_target``
используется исключительно для расчёта метрик, но не для обучения,
отбора признаков или подбора гиперпараметров.
"""
from __future__ import annotations

import numpy as np

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def _as_2d(values) -> np.ndarray:
    """Привести матрицу признаков к двумерному float-массиву."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _as_1d(values) -> np.ndarray:
    """Привести вектор меток к одномерному массиву (без потери типа)."""
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучить soft-voting ансамбль и вернуть прогнозы и метрики на test."""
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y_train_raw = _as_1d(data.train_target)
    y_test_raw = _as_1d(data.test_target)

    classes = np.unique(y_train_raw)
    if classes.size != 2:
        raise ValueError(
            "Ожидается бинарная классификация: в train должно быть ровно два класса"
        )
    # sklearn сортирует классы по возрастанию; положительным считаем больший.
    positive = classes[1]
    y_train = (y_train_raw == positive).astype(int)
    y_test = (y_test_raw == positive).astype(int)

    seed = int(data.random_seed)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features=0.33,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )

    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft",
    )
    model.fit(x_train, y_train)

    probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    positive_mask = (probabilities >= 0.5).astype(int)
    # Возвращаем предсказания в исходном пространстве меток.
    predictions = np.where(positive_mask == 1, positive, classes[0])

    return EnsembleResult(
        model_name="SoftVoting(RandomForest + GradientBoosting + ExtraTrees)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, positive_mask)),
        precision=float(precision_score(y_test, positive_mask, zero_division=0)),
        recall=float(recall_score(y_test, positive_mask, zero_division=0)),
        f1=float(f1_score(y_test, positive_mask, zero_division=0)),
    )
