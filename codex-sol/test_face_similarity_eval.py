#!/usr/bin/env python3
"""无需第三方测试框架的核心正确性测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from face_similarity_eval import evaluate_similarity, extract_pairs, pair_counts, plot_results, save_pairs


def _fixture() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(20260905)
    features = rng.normal(size=(37, 16)).astype(np.float32)
    ids = np.array([0] * 4 + [1] * 5 + [2] * 3 + list(range(3, 28)), dtype=np.int64)
    paths = [f"image_{index}.jpg" for index in range(len(ids))]
    return features, ids, paths


def test_counts_and_scores() -> None:
    features, ids, _ = _fixture()
    result = evaluate_similarity(features, ids, block_size=7, bins=2048, devices=["cpu"], collect_scores=True)
    normalised = features / np.linalg.norm(features, axis=1, keepdims=True)
    matrix = normalised @ normalised.T
    row, col = np.triu_indices(len(ids), k=1)
    positive_mask = ids[row] == ids[col]
    expected_positive = matrix[row[positive_mask], col[positive_mask]]
    expected_negative = matrix[row[~positive_mask], col[~positive_mask]]
    total, positive, negative = pair_counts(ids)
    assert (total, positive, negative) == (len(row), len(expected_positive), len(expected_negative))
    assert int(result.positive_hist.sum()) == positive
    assert int(result.negative_hist.sum()) == negative
    np.testing.assert_allclose(np.sort(result.positive_scores), np.sort(expected_positive), atol=1e-6)
    np.testing.assert_allclose(np.sort(result.negative_scores), np.sort(expected_negative), atol=1e-6)


def test_metrics_match_exact_definition() -> None:
    features, ids, _ = _fixture()
    result = evaluate_similarity(features, ids, block_size=8, devices=["cpu"], collect_scores=True)
    for row in result.metrics_at_fpir((0.0, 0.01, 0.1, 1.0)):
        threshold = row["threshold"]
        expected_fp = np.count_nonzero(result.negative_scores > threshold)
        expected_tp = np.count_nonzero(result.positive_scores > threshold)
        assert int(row["false_positives"]) == expected_fp
        assert int(row["true_positives"]) == expected_tp
        assert row["fpir"] <= row["fpir_target"] + 1e-12
    _, curve_fpir, curve_tpir = result.curve()
    assert np.all(np.diff(curve_fpir) >= -1e-12)
    assert np.all(np.diff(curve_tpir) >= -1e-12)


def test_pair_extraction_and_outputs() -> None:
    features, ids, paths = _fixture()
    rows = extract_pairs(features, ids, paths, threshold=0.2, mode="above", pair_type="negative", block_size=6, limit=20)
    assert len(rows) <= 20
    assert all(not row["is_positive"] and row["index_i"] < row["index_j"] for row in rows)
    assert all(row["similarity"] >= 0.2 for row in rows)
    result = evaluate_similarity(features, ids, block_size=9, bins=512, devices=["cpu"], collect_scores=False)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        save_pairs(rows, output / "pairs.parquet")
        distribution, curve = plot_results(result, output, "font/SourceHanSansSC-Normal.otf")
        assert (output / "pairs.parquet").stat().st_size > 0
        assert distribution.stat().st_size > 0
        assert curve.stat().st_size > 0


def main() -> None:
    test_counts_and_scores()
    test_metrics_match_exact_definition()
    test_pair_extraction_and_outputs()
    print("全部测试通过")


if __name__ == "__main__":
    main()
