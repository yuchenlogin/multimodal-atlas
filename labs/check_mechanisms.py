#!/usr/bin/env python3
"""Execute small CPU mechanism checks and save inspectable, constructed data.

Run from any directory with the repository's Python environment:
    .venv/bin/python labs/check_mechanisms.py

These are numerical checks, not trained-model or paper-result reproductions.
No network, accelerator, model checkpoint, or external dataset is used.
Only sources/mechanism-checks.json and sources/raw/check-*.csv are generated.
Each random component has a fixed, independent SeedSequence spawn key.
"""

from __future__ import annotations

import sys
import csv
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "sources" / "raw"
OUTPUT = ROOT / "sources" / "mechanism-checks.json"
DTYPE = np.float64


def random_stream(base_seed: int, topic: int, stream: int = 0):
    """Explicit keys keep one check independent of all other checks' draws."""
    sequence = np.random.SeedSequence(base_seed, spawn_key=(topic, stream))
    rng = np.random.Generator(np.random.PCG64(sequence))
    config = {
        "base_seed": base_seed,
        "spawn_key": [topic, stream],
        "bit_generator": "PCG64",
        "randomness_used": True,
    }
    return rng, config


def deterministic_stream(base_seed: int, topic: int):
    """Reserve a key without falsely claiming a deterministic fixture is random."""
    return {
        "base_seed": base_seed,
        "reserved_spawn_key": [topic, 0],
        "randomness_used": False,
    }


class RawWriter:
    def __init__(self):
        self.manifest = []

    def write(self, filename: str, columns: list[str], rows):
        path = RAW_DIR / filename
        count = 0
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
        relative = path.relative_to(ROOT).as_posix()
        self.manifest.append({
            "path": relative,
            "format": "CSV with header, UTF-8",
            "rows": count,
            "columns": columns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        return relative


def check_close(actual, expected, tolerance: float, label: str):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f"{label}: max absolute error {error} > {tolerance}")
    return error


def log_softmax(logits):
    logits = np.asarray(logits, dtype=DTYPE)
    maximum = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sigmoid(value):
    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=DTYPE)))


def finite_difference(function, point, step: float = 1e-6):
    point = np.asarray(point, dtype=DTYPE)
    result = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def rotate_pairs(values, positions, theta: float = 10000.0):
    """RoPE on adjacent feature pairs; last dimension must be even."""
    values = np.asarray(values, dtype=DTYPE)
    dimension = values.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE check requires an even head dimension")
    frequency = theta ** (-np.arange(0, dimension, 2, dtype=DTYPE) / dimension)
    angle = np.asarray(positions, dtype=DTYPE)[..., None] * frequency
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.empty_like(values)
    result[..., 0::2] = values[..., 0::2] * cosine - values[..., 1::2] * sine
    result[..., 1::2] = values[..., 0::2] * sine + values[..., 1::2] * cosine
    return result


def check_rope_cache(base_seed: int, raw: RawWriter):
    rng, seed = random_stream(base_seed, 0)
    length, dimension, offset = 7, 8, 11
    q, k, v = (rng.normal(size=(length, dimension)) for _ in range(3))
    positions = np.arange(length)
    qr = rotate_pairs(q, positions)
    kr = rotate_pairs(k, positions)
    score = qr @ kr.T
    shifted = rotate_pairs(q, positions + offset) @ rotate_pairs(k, positions + offset).T
    invariance_error = check_close(shifted, score, 1e-12, "RoPE relative-position invariance")
    masked_score = np.where(
        np.tril(np.ones((length, length), dtype=bool)), score / math.sqrt(dimension), -np.inf
    )
    full = np.exp(log_softmax(masked_score)) @ v
    cached, wrong_position = [], []
    for t in range(length):
        # K is cached after its one position-dependent rotation; V is not rotated.
        key_cache, value_cache = kr[:t + 1], v[:t + 1]
        query = rotate_pairs(q[t:t + 1], np.array([t]))
        weights = np.exp(log_softmax(query @ key_cache.T / math.sqrt(dimension)))
        cached.append((weights @ value_cache)[0])
        # Negative control: incorrectly treat every new query as position zero.
        wrong_query = rotate_pairs(q[t:t + 1], np.array([0]))
        wrong_weights = np.exp(log_softmax(wrong_query @ key_cache.T / math.sqrt(dimension)))
        wrong_position.append((wrong_weights @ value_cache)[0])
    cached, wrong_position = np.asarray(cached), np.asarray(wrong_position)
    cache_error = check_close(cached, full, 1e-12, "causal full versus cached attention")
    wrong_error = float(np.max(np.abs(wrong_position - full)))
    files = [raw.write(
        "check-00-rope-cache.csv",
        ["position", "feature", "q", "k", "v", "rotated_q", "rotated_k",
         "full_attention", "cached_attention", "wrong_query_position_attention"],
        ({"position": t, "feature": j, "q": q[t, j], "k": k[t, j], "v": v[t, j],
          "rotated_q": qr[t, j], "rotated_k": kr[t, j], "full_attention": full[t, j],
          "cached_attention": cached[t, j],
          "wrong_query_position_attention": wrong_position[t, j]}
         for t in range(length) for j in range(dimension)),
    )]
    files.append(raw.write(
        "check-00-relative-scores.csv", ["query_position", "key_position", "score", "shifted_score"],
        ({"query_position": i, "key_position": j, "score": score[i, j],
          "shifted_score": shifted[i, j]}
         for i in range(length) for j in range(length)),
    ))
    return {
        "title": "RoPE relative invariance and causal KV-cache equivalence",
        "kind": "executed CPU mechanism check",
        "config": {"rng": seed, "dtype": "float64", "tokens": length,
                   "head_dimension": dimension, "theta": 10000, "position_shift": offset,
                   "mask": "lower triangular including diagonal", "dropout": 0,
                   "cache": "rotated K, unrotated V; absolute query positions"},
        "results": {"relative_score_max_abs_error": invariance_error,
                    "cached_output_max_abs_error": cache_error,
                    "wrong_query_position_max_abs_error": wrong_error,
                    "tolerance": 1e-12},
        "raw_data": files,
        "limits": "One synthetic attention head; no trained transformer, speed, or GPU-memory claim.",
        "passed": True,
    }


def check_contrastive_gradients(base_seed: int, raw: RawWriter):
    rng, seed = random_stream(base_seed, 1)
    n, step, bias = 4, 1e-6, -0.7
    logits = rng.normal(scale=0.7, size=(n, n))
    labels = 2 * np.eye(n) - 1

    def clip_loss(matrix):
        return float(-np.diag(log_softmax(matrix)).mean())

    def siglip_loss(matrix, shared_bias=bias):
        return float(np.logaddexp(0.0, -labels * (matrix + shared_bias)).sum() / n)

    clip_grad = (np.exp(log_softmax(logits)) - np.eye(n)) / n
    sig_grad = -labels * sigmoid(-labels * (logits + bias)) / n
    clip_numeric = finite_difference(clip_loss, logits, step)
    sig_numeric = finite_difference(siglip_loss, logits, step)
    clip_error = check_close(clip_numeric, clip_grad, 1e-7, "CLIP logit gradient")
    sig_error = check_close(sig_numeric, sig_grad, 1e-7, "SigLIP logit gradient")
    bias_numeric = (siglip_loss(logits, bias + step) - siglip_loss(logits, bias - step)) / (2 * step)
    bias_error = check_close(bias_numeric, sig_grad.sum(), 1e-7, "SigLIP shared bias gradient")
    baseline = []
    for size in (2, 8, 64):
        pair_labels = 2 * np.eye(size) - 1
        zero_clip_grad = (np.full((size, size), 1 / size) - np.eye(size)) / size
        zero_sig_grad = -pair_labels / (2 * size)
        calibrated_bias = -math.log(size - 1)
        calibrated_bias_grad = float((-pair_labels * sigmoid(-pair_labels * calibrated_bias) / size).sum())
        clip_norm, sig_norm = float(np.linalg.norm(zero_clip_grad)), float(np.linalg.norm(zero_sig_grad))
        check_close(clip_norm, math.sqrt(size - 1) / size, 1e-12, "CLIP zero-logit norm")
        check_close(sig_norm, 0.5, 1e-12, "SigLIP zero-logit norm")
        check_close(zero_sig_grad.sum(), (size - 2) / 2, 1e-12, "SigLIP zero-bias imbalance")
        check_close(calibrated_bias_grad, 0, 1e-12, "prior-calibrated bias gradient")
        baseline.append({
            "N": size, "clip_loss": math.log(size), "siglip_loss": size * math.log(2),
            "clip_logit_gradient_frobenius_norm": clip_norm,
            "siglip_logit_gradient_frobenius_norm": sig_norm,
            "siglip_zero_bias_gradient": float(zero_sig_grad.sum()),
            "prior_calibrated_bias": calibrated_bias,
            "prior_calibrated_bias_gradient": calibrated_bias_grad,
        })
    gradient_rows = []
    for objective, analytical, numerical in (
        ("CLIP_one_direction_mean", clip_grad, clip_numeric),
        ("SigLIP_pair_sum_divided_by_N", sig_grad, sig_numeric),
    ):
        for i, j in np.ndindex(logits.shape):
            gradient_rows.append({"objective": objective, "row": i, "column": j,
                                  "logit": logits[i, j], "pair_label": labels[i, j],
                                  "analytic_gradient": analytical[i, j],
                                  "finite_difference_gradient": numerical[i, j],
                                  "absolute_error": abs(analytical[i, j] - numerical[i, j])})
    files = [raw.write("check-01-logit-gradients.csv", list(gradient_rows[0]), gradient_rows),
             raw.write("check-01-zero-logit-baselines.csv", list(baseline[0]), baseline)]
    return {
        "title": "CLIP and SigLIP logit/bias gradients",
        "kind": "executed CPU finite differences plus analytic baselines",
        "config": {"rng": seed, "dtype": "float64", "N": n,
                   "logit_distribution": "Normal(0,0.7^2)", "shared_bias": bias,
                   "finite_difference": "central", "finite_difference_step": step,
                   "CLIP_normalization": "one-direction cross entropy averaged over N rows",
                   "SigLIP_normalization": "all N^2 pair losses summed, divided by N",
                   "baseline_N": [2, 8, 64]},
        "results": {"clip_loss": clip_loss(logits), "siglip_loss": siglip_loss(logits),
                    "clip_max_gradient_error": clip_error, "siglip_max_gradient_error": sig_error,
                    "siglip_bias_analytic_gradient": float(sig_grad.sum()),
                    "siglip_bias_finite_difference": bias_numeric, "bias_gradient_error": bias_error,
                    "zero_logit_baselines": baseline, "tolerance": 1e-7},
        "raw_data": files,
        "limits": "Gradients are with respect to logits and a shared bias, not encoder parameters or model quality.",
        "passed": True,
    }


def collapse_ctc(path):
    result, previous = [], None
    for token in path:
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return tuple(result)


def ctc_forward_probability(probabilities, target):
    """Independent forward-DP check; zero is blank, repeats need a blank."""
    extended = [0]
    for token in target:
        extended.extend([token, 0])
    alpha = np.zeros(len(extended), dtype=DTYPE)
    alpha[0] = probabilities[0, 0]
    if target:
        alpha[1] = probabilities[0, target[0]]
    for frame in probabilities[1:]:
        updated = np.zeros_like(alpha)
        for state, token in enumerate(extended):
            total = alpha[state]
            if state > 0:
                total += alpha[state - 1]
            if state > 1 and token != 0 and token != extended[state - 2]:
                total += alpha[state - 2]
            updated[state] = total * frame[token]
        alpha = updated
    return float(alpha[-1] + (alpha[-2] if target else 0.0))


def check_ctc_paths(base_seed: int, raw: RawWriter):
    symbols = ["blank", "A", "B"]
    probabilities = np.array([[0.1, 0.6, 0.3], [0.2, 0.5, 0.3], [0.45, 0.35, 0.2]], dtype=DTYPE)
    check_close(probabilities.sum(axis=1), 1, 1e-12, "CTC categorical normalization")
    all_paths, mass, counts = [], {}, {}
    for path in itertools.product(range(3), repeat=3):
        collapsed = collapse_ctc(path)
        probability = float(np.prod(probabilities[np.arange(3), path]))
        mass[collapsed] = mass.get(collapsed, 0.0) + probability
        counts[collapsed] = counts.get(collapsed, 0) + 1
        all_paths.append({"path": "|".join(symbols[x] for x in path),
                          "collapsed": "|".join(symbols[x] for x in collapsed),
                          "path_probability": probability})
    check_close(sum(mass.values()), 1, 1e-12, "CTC exhaustive path mass")
    targets = []
    for target in ((1,), (1, 1), (1, 2)):
        forward = ctc_forward_probability(probabilities, target)
        error = check_close(forward, mass.get(target, 0.0), 1e-12, "CTC forward versus enumeration")
        targets.append({"target": "|".join(symbols[x] for x in target),
                        "enumerated_paths": counts.get(target, 0),
                        "enumerated_probability": mass.get(target, 0.0),
                        "forward_probability": forward, "absolute_error": error})
    greedy = tuple(probabilities.argmax(axis=1))
    emission_rows = [{"frame": t, "symbol": symbols[j], "probability": probabilities[t, j]}
                     for t in range(3) for j in range(3)]
    files = [raw.write("check-02-ctc-paths.csv", list(all_paths[0]), all_paths),
             raw.write("check-02-ctc-emissions.csv", list(emission_rows[0]), emission_rows)]
    return {
        "title": "CTC merge-then-remove-blank path accounting",
        "kind": "executed exhaustive enumeration and forward dynamic program",
        "config": {"rng": deterministic_stream(base_seed, 2), "dtype": "float64",
                   "frames": 3, "symbols": symbols, "blank_id": 0,
                   "collapse_order": "merge consecutive identical labels, then remove blank"},
        "results": {"total_paths": len(all_paths), "total_probability": sum(mass.values()),
                    "target_results": targets, "greedy_path": [symbols[x] for x in greedy],
                    "greedy_collapsed": [symbols[x] for x in collapse_ctc(greedy)],
                    "greedy_path_probability": float(np.prod(probabilities[np.arange(3), greedy])),
                    "A_blank_A_collapsed": [symbols[x] for x in collapse_ctc((1, 0, 1))],
                    "A_A_A_collapsed": [symbols[x] for x in collapse_ctc((1, 1, 1))],
                    "tolerance": 1e-12},
        "raw_data": files,
        "limits": "Three independent categorical frames; no acoustic model, Whisper, codec, or recognition benchmark.",
        "passed": True,
    }


def check_diffusion_parameters(base_seed: int, raw: RawWriter):
    rng, seed = random_stream(base_seed, 5)
    shape, alphas = (4, 8), [0.02, 0.1, 0.5, 0.9, 0.98]
    x0, epsilon = rng.normal(size=shape), rng.normal(size=shape)
    perturbation = 0.01 * rng.normal(size=shape)
    summaries, rows = [], []
    for alpha in alphas:
        a, b = math.sqrt(alpha), math.sqrt(1 - alpha)
        xt = a * x0 + b * epsilon
        velocity = a * epsilon - b * x0
        x0_from_epsilon = (xt - b * epsilon) / a
        epsilon_from_x0 = (xt - a * x0) / b
        x0_from_v = a * xt - b * velocity
        epsilon_from_v = b * xt + a * velocity
        perturbed_x0 = (xt - b * (epsilon + perturbation)) / a
        errors = {
            "x0_from_epsilon_max_error": check_close(x0_from_epsilon, x0, 1e-12, "epsilon to x0"),
            "epsilon_from_x0_max_error": check_close(epsilon_from_x0, epsilon, 1e-12, "x0 to epsilon"),
            "x0_from_v_max_error": check_close(x0_from_v, x0, 1e-12, "v to x0"),
            "epsilon_from_v_max_error": check_close(epsilon_from_v, epsilon, 1e-12, "v to epsilon"),
        }
        amplification = float(np.linalg.norm(perturbed_x0 - x0) / np.linalg.norm(perturbation))
        amplification_error = check_close(amplification, b / a, 1e-12, "noise-error amplification")
        summaries.append({"cumulative_alpha": alpha, **errors,
                          "measured_error_amplification": amplification,
                          "analytic_error_amplification": b / a,
                          "amplification_absolute_error": amplification_error})
        for i, j in np.ndindex(shape):
            rows.append({"cumulative_alpha": alpha, "sample": i, "feature": j,
                         "x0": x0[i, j], "epsilon": epsilon[i, j], "xt": xt[i, j],
                         "v": velocity[i, j], "x0_from_epsilon": x0_from_epsilon[i, j],
                         "epsilon_from_x0": epsilon_from_x0[i, j], "x0_from_v": x0_from_v[i, j],
                         "epsilon_from_v": epsilon_from_v[i, j],
                         "epsilon_perturbation": perturbation[i, j],
                         "perturbed_x0": perturbed_x0[i, j]})
    files = [raw.write("check-05-parameterization.csv", list(rows[0]), rows)]
    return {
        "title": "Diffusion epsilon/x0/v inversion and fixed-error amplification",
        "kind": "executed CPU algebra check",
        "config": {"rng": seed, "dtype": "float64", "shape": list(shape),
                   "cumulative_alphas": alphas, "x0_and_epsilon": "independent standard normals",
                   "epsilon_perturbation": "0.01 times independent standard normal; shared across alpha",
                   "xt": "sqrt(alpha)*x0 + sqrt(1-alpha)*epsilon",
                   "v": "sqrt(alpha)*epsilon - sqrt(1-alpha)*x0"},
        "results": {"by_alpha": summaries, "tolerance": 1e-12},
        "raw_data": files,
        "limits": "Known synthetic vectors and exact targets, not a trained denoiser or image-generation quality test.",
        "passed": True,
    }


def online_attention(scores, values, block_size: int):
    query_count, key_count = scores.shape
    maximum = np.full(query_count, -np.inf, dtype=DTYPE)
    denominator = np.zeros(query_count, dtype=DTYPE)
    numerator = np.zeros((query_count, values.shape[1]), dtype=DTYPE)
    for start in range(0, key_count, block_size):
        stop = min(start + block_size, key_count)
        tile = scores[:, start:stop]
        updated_maximum = np.maximum(maximum, tile.max(axis=1))
        old_rescale = np.exp(maximum - updated_maximum)
        tile_exp = np.exp(tile - updated_maximum[:, None])
        denominator = old_rescale * denominator + tile_exp.sum(axis=1)
        numerator = old_rescale[:, None] * numerator + tile_exp @ values[start:stop]
        maximum = updated_maximum
    return numerator / denominator[:, None]


def check_online_softmax(base_seed: int, raw: RawWriter):
    rng, seed = random_stream(base_seed, 8)
    nq, nk, dimension, value_dimension = 7, 19, 8, 5
    q = rng.normal(size=(nq, dimension))
    k = rng.normal(size=(nk, dimension))
    v = rng.normal(size=(nk, value_dimension))
    offsets = 1000 + np.arange(nq, dtype=DTYPE)
    scores = q @ k.T / math.sqrt(dimension) + offsets[:, None]
    full = np.exp(log_softmax(scores)) @ v
    summaries, output_rows = [], []
    for block_size in (1, 4, 7, 19):
        online = online_attention(scores, v, block_size)
        error = check_close(online, full, 1e-12, "online softmax versus full attention")
        summaries.append({"block_size": block_size, "output_max_abs_error": error,
                          "score_tile_entries": nq * min(block_size, nk),
                          "score_tile_bytes_float64": nq * min(block_size, nk) * 8})
        for i, j in np.ndindex(full.shape):
            output_rows.append({"block_size": block_size, "query": i, "value_feature": j,
                                "full_output": full[i, j], "online_output": online[i, j],
                                "absolute_error": abs(online[i, j] - full[i, j])})
    files = []
    for name, array in (("q", q), ("k", k), ("v", v)):
        files.append(raw.write(
            f"check-08-{name}.csv", ["position", "feature", "value"],
            ({"position": i, "feature": j, "value": array[i, j]}
             for i, j in np.ndindex(array.shape)),
        ))
    files.append(raw.write(
        "check-08-scores.csv", ["query", "key", "row_offset", "score"],
        ({"query": i, "key": j, "row_offset": offsets[i], "score": scores[i, j]}
         for i, j in np.ndindex(scores.shape)),
    ))
    files.append(raw.write("check-08-online-outputs.csv", list(output_rows[0]), output_rows))
    return {
        "title": "Stable blockwise online softmax versus full attention",
        "kind": "executed CPU numerical-equivalence check",
        "config": {"rng": seed, "dtype": "float64", "queries": nq, "keys": nk,
                   "head_dimension": dimension, "value_dimension": value_dimension,
                   "block_sizes": [1, 4, 7, 19], "score": "QK^T/sqrt(d) + row_offset",
                   "row_offsets": offsets.tolist(), "mask": "none", "dropout": 0},
        "results": {"by_block_size": summaries,
                    "full_score_entries": nq * nk, "full_score_bytes_float64": nq * nk * 8,
                    "accumulator_entries": nq * (value_dimension + 2), "tolerance": 1e-12},
        "raw_data": files,
        "limits": "The full score matrix is deliberately retained as a reference fixture. Tile sizes are analytic storage counts, not measured peak RAM, FlashAttention throughput, or GPU speedups.",
        "passed": True,
    }


def check_dpo_lengths(base_seed: int, raw: RawWriter):
    symbols = ["chosen_token", "rejected_token", "other", "EOS"]
    reference_p = np.array([0.2, 0.2, 0.5, 0.1], dtype=DTYPE)
    policy_p = np.array([0.2 * math.exp(0.02), 0.2 * math.exp(0.01), 0.0, 0.1], dtype=DTYPE)
    policy_p[2] = 1 - policy_p[[0, 1, 3]].sum()
    if np.any(policy_p <= 0) or np.any(reference_p <= 0):
        raise AssertionError("DPO fixture needs strictly positive categorical probabilities")
    check_close(policy_p.sum(), 1, 1e-12, "DPO policy normalization")
    check_close(reference_p.sum(), 1, 1e-12, "DPO reference normalization")
    policy_logits, reference_logp = np.log(policy_p), np.log(reference_p)
    chosen, rejected = np.array([0] * 5 + [3]), np.array([1] * 20 + [3])
    chosen_count, rejected_count = (np.bincount(x, minlength=4) for x in (chosen, rejected))
    beta, step = 0.5, 1e-6

    def margin(logits, normalization):
        ratio = log_softmax(logits) - reference_logp
        if normalization == "sequence_sum":
            return float(ratio[chosen].sum() - ratio[rejected].sum())
        return float(ratio[chosen].mean() - ratio[rejected].mean())

    results, gradient_rows = [], []
    for normalization in ("sequence_sum", "answer_token_mean"):
        delta = margin(policy_logits, normalization)
        loss = float(np.logaddexp(0.0, -beta * delta))
        derivative = float(-beta * sigmoid(-beta * delta))
        if normalization == "sequence_sum":
            margin_grad = chosen_count - rejected_count - (len(chosen) - len(rejected)) * policy_p
        else:
            margin_grad = chosen_count / len(chosen) - rejected_count / len(rejected)
        analytic = derivative * margin_grad
        numeric = finite_difference(
            lambda logits: float(np.logaddexp(0.0, -beta * margin(logits, normalization))),
            policy_logits, step,
        )
        gradient_error = check_close(numeric, analytic, 1e-7, "DPO policy-logit gradient")
        derivative_numeric = float((np.logaddexp(0.0, -beta * (delta + step)) -
                                    np.logaddexp(0.0, -beta * (delta - step))) / (2 * step))
        derivative_error = check_close(derivative_numeric, derivative, 1e-7, "DPO margin derivative")
        results.append({"normalization": normalization, "margin": delta, "loss": loss,
                        "d_loss_d_margin": derivative, "margin_derivative_finite_difference": derivative_numeric,
                        "margin_derivative_error": derivative_error,
                        "max_policy_logit_gradient_error": gradient_error})
        for j in range(4):
            gradient_rows.append({"normalization": normalization, "symbol": symbols[j],
                                  "policy_logit": policy_logits[j], "analytic_gradient": analytic[j],
                                  "finite_difference_gradient": numeric[j],
                                  "absolute_error": abs(analytic[j] - numeric[j])})
    if not results[0]["margin"] < 0 < results[1]["margin"]:
        raise AssertionError("Length-normalization fixture must reverse the margin's sign")
    token_rows = []
    policy_logp = log_softmax(policy_logits)
    for answer_name, answer in (("chosen", chosen), ("rejected", rejected)):
        for position, token in enumerate(answer):
            token_rows.append({"answer": answer_name, "position": position, "symbol": symbols[token],
                               "is_EOS": token == 3, "policy_log_probability": policy_logp[token],
                               "reference_log_probability": reference_logp[token],
                               "log_ratio": policy_logp[token] - reference_logp[token]})
    distribution_rows = [{"symbol": symbols[j], "reference_probability": reference_p[j],
                          "policy_probability": policy_p[j]} for j in range(4)]
    files = [raw.write("check-09-token-logprobs.csv", list(token_rows[0]), token_rows),
             raw.write("check-09-distributions.csv", list(distribution_rows[0]), distribution_rows),
             raw.write("check-09-gradients.csv", list(gradient_rows[0]), gradient_rows)]
    return {
        "title": "Legal autoregressive log probabilities: sequence sum versus token mean",
        "kind": "executed deterministic categorical fixture and finite differences",
        "config": {"rng": deterministic_stream(base_seed, 9), "dtype": "float64", "vocabulary": symbols,
                   "model": "same normalized categorical distribution at every context; terminate on EOS",
                   "chosen_content_tokens": 5, "rejected_content_tokens": 20,
                   "answer_token_counts_including_EOS": [len(chosen), len(rejected)],
                   "EOS_policy": "included in both answer scores; identical EOS probability in policy and reference",
                   "reference": "frozen", "beta": beta, "finite_difference_step": step,
                   "token_mean": "each policy/reference answer log probability divided by its own valid-answer-token count"},
        "results": {"by_normalization": results, "margin_sign_reversed": True, "tolerance": 1e-7},
        "raw_data": files,
        "limits": "A fixed categorical toy with valid complete EOS-terminated sequence probabilities; no DPO training, preference-quality, or visual-grounding claim.",
        "passed": True,
    }


def bootstrap_summary(values):
    return {"mean": float(values.mean()), "standard_deviation": float(values.std(ddof=1)),
            "ci95_percentile": np.quantile(values, [0.025, 0.975], method="linear").tolist()}


def check_cluster_bootstrap(base_seed: int, draws: int, raw: RawWriter):
    item_rng, item_seed = random_stream(base_seed, 10, 0)
    cluster_rng, cluster_seed = random_stream(base_seed, 10, 1)
    # Same 80/10/20/90 record counts as the teaching example; a NEW clustered fixture.
    cluster_scores = [(1, 1)] * 8 + [(1, 0)] + [(0, 1)] * 2 + [(0, 0)] * 9
    rows = []
    for cluster_id, (score_a, score_b) in enumerate(cluster_scores):
        for within_cluster in range(10):
            rows.append({"item_id": f"c{cluster_id:02d}-i{within_cluster:02d}",
                         "cluster_id": f"c{cluster_id:02d}", "score_A": score_a, "score_B": score_b,
                         "difference_B_minus_A": score_b - score_a})
    differences = np.array([row["difference_B_minus_A"] for row in rows], dtype=DTYPE)
    cluster_differences = np.array([b - a for a, b in cluster_scores], dtype=DTYPE)
    item_draws = item_rng.choice(differences, (draws, len(differences)), replace=True).mean(axis=1)
    cluster_draws = cluster_rng.choice(
        cluster_differences, (draws, len(cluster_differences)), replace=True
    ).mean(axis=1)
    counts = {"both_correct": sum(a == b == 1 for a, b in cluster_scores) * 10,
              "A_only": sum(a == 1 and b == 0 for a, b in cluster_scores) * 10,
              "B_only": sum(a == 0 and b == 1 for a, b in cluster_scores) * 10,
              "both_wrong": sum(a == b == 0 for a, b in cluster_scores) * 10}
    check_close(differences.mean(), 0.05, 1e-12, "paired point estimate")
    check_close(cluster_differences.mean(), differences.mean(), 1e-12, "equal cluster point estimate")
    draw_rows = [{"draw_id": i, "item_bootstrap_difference": item_draws[i],
                  "cluster_bootstrap_difference": cluster_draws[i]} for i in range(draws)]
    files = [raw.write("check-10-paired-items.csv", list(rows[0]), rows),
             raw.write("check-10-bootstrap-draws.csv", list(draw_rows[0]), draw_rows)]
    return {
        "title": "Identical paired record counts, different independent-unit assumptions",
        "kind": "executed bootstrap on explicitly constructed clustered outcomes",
        "config": {"rng_item_resampling": item_seed, "rng_cluster_resampling": cluster_seed,
                   "items": 200, "clusters": 20, "items_per_cluster": 10, "bootstrap_draws": draws,
                   "within_cluster": "all ten records have the same A/B outcome",
                   "item_bootstrap": "resample 200 paired records; ignores constructed clustering",
                   "cluster_bootstrap": "resample 20 clusters and retain all ten paired records per cluster; equal cluster sizes",
                   "quantile_method": "NumPy linear percentile", "interval_level": 0.95,
                   "unit": "B minus A accuracy fraction; multiply by 100 for percentage points"},
        "results": {"record_counts": counts, "accuracy_A": 0.45, "accuracy_B": 0.5,
                    "paired_difference": float(differences.mean()),
                    "item_bootstrap": bootstrap_summary(item_draws),
                    "cluster_bootstrap": bootstrap_summary(cluster_draws),
                    "item_based_plugin_SE": float(differences.std(ddof=1) / math.sqrt(len(differences))),
                    "cluster_based_plugin_SE": float(cluster_differences.std(ddof=1) / math.sqrt(len(cluster_differences)))},
        "raw_data": files,
        "limits": "Constructed perfect within-cluster dependence. This new fixture changes the assumed sampling structure; it does not retroactively prove the original 200 independent-item toy invalid, and contains no model calls.",
        "passed": True,
    }


def fit_ols(design, target, coefficient_names):
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    return {"coefficients": dict(zip(coefficient_names, coefficients.tolist())),
            "rank": int(rank), "design_condition_number": float(np.linalg.cond(design)),
            "residual_mean_squared_error": float(np.mean(residuals ** 2))}


def check_adjusted_ols(base_seed: int, raw: RawWriter):
    observed_rng, observed_seed = random_stream(base_seed, 12, 0)
    intervention_rng, intervention_seed = random_stream(base_seed, 12, 1)
    sample_count = 10000
    background = observed_rng.normal(size=sample_count)
    action_noise = observed_rng.normal(size=sample_count)
    response_noise = 0.1 * observed_rng.normal(size=sample_count)
    action = background + 0.3 * action_noise
    response = 0.2 * action + 0.8 * background + response_noise
    randomized_background = intervention_rng.normal(size=sample_count)
    randomized_action = intervention_rng.normal(size=sample_count)
    randomized_noise = 0.1 * intervention_rng.normal(size=sample_count)
    randomized_response = 0.2 * randomized_action + 0.8 * randomized_background + randomized_noise
    ones = np.ones(sample_count)
    naive = fit_ols(np.column_stack([ones, action]), response, ["intercept", "action"])
    adjusted = fit_ols(np.column_stack([ones, action, background]), response,
                       ["intercept", "action", "background"])
    randomized = fit_ols(np.column_stack([ones, randomized_action]), randomized_response,
                         ["intercept", "action"])
    closed_naive = float(np.cov(action, response, ddof=0)[0, 1] / np.var(action))
    check_close(naive["coefficients"]["action"], closed_naive, 1e-12, "OLS with intercept versus covariance formula")
    columns = ["sample_id", "regime", "background", "exogenous_action_noise", "action", "response_noise", "response"]
    files = [raw.write(
        "check-12-observational.csv", columns,
        ({"sample_id": i, "regime": "observational", "background": background[i],
          "exogenous_action_noise": action_noise[i], "action": action[i],
          "response_noise": response_noise[i], "response": response[i]}
         for i in range(sample_count)),
    )]
    files.append(raw.write(
        "check-12-randomized.csv", columns,
        ({"sample_id": i, "regime": "randomized", "background": randomized_background[i],
          "exogenous_action_noise": randomized_action[i], "action": randomized_action[i],
          "response_noise": randomized_noise[i], "response": randomized_response[i]}
         for i in range(sample_count)),
    ))
    return {
        "title": "Confounded naive OLS, observed-confounder adjustment, and randomized action",
        "kind": "executed OLS on new, independently seeded synthetic samples",
        "config": {"rng_observational": observed_seed, "rng_randomized": intervention_seed,
                   "dtype": "float64", "samples_per_regime": sample_count,
                   "observational_action": "a=c+0.3*n; c,n independent standard normal",
                   "randomized_action": "a standard normal, independent of new background c",
                   "response": "y=0.2*a+0.8*c+e; independent e~Normal(0,0.1^2)",
                   "OLS": "includes intercept; NumPy lstsq rcond=None",
                   "adjustment_assumptions": "observed c, correct linear specification, independent structural noise, nonzero action variation given c",
                   "separation_from_old_probe": "does not read sources/probes.json or reuse its RNG state/samples"},
        "results": {"population_naive_OLS_coefficient": 0.2 + 0.8 / 1.09,
                    "structural_action_coefficient": 0.2,
                    "observational_naive": naive, "observational_adjusted": adjusted,
                    "randomized_unadjusted": randomized},
        "raw_data": files,
        "limits": "Known scalar structural equations with observed confounder; no trained dynamics, temporal rollout, planning policy, robot, or real causal discovery claim.",
        "passed": True,
    }


def main():
    if len(sys.argv) != 1:
        raise SystemExit("Publication fixtures use fixed seed=336 and 4000 draws; this generator accepts no overrides.")
    seed, bootstrap_draws = 336, 4000
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RawWriter()
    checks = {
        "00": check_rope_cache(seed, raw),
        "01": check_contrastive_gradients(seed, raw),
        "02": check_ctc_paths(seed, raw),
        "05": check_diffusion_parameters(seed, raw),
        "08": check_online_softmax(seed, raw),
        "09": check_dpo_lengths(seed, raw),
        "10": check_cluster_bootstrap(seed, bootstrap_draws, raw),
        "12": check_adjusted_ols(seed, raw),
    }
    output = {
        "schema_version": "1.0",
        "scope": "Executed CPU-only checks on synthetic vectors, legal categorical distributions, constructed outcomes and known structural equations. No foundation-model or paper-result reproduction.",
        "generator": {"path": "labs/check_mechanisms.py",
                      "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "command": "python labs/check_mechanisms.py",
                      "base_seed": seed,
                      "environment": {"python": platform.python_version(), "numpy": np.__version__},
                      "randomness": "independent PCG64 SeedSequence(base_seed, spawn_key=(topic,stream)); deterministic fixtures explicitly marked"},
        "checks": checks,
        "raw_files": raw.manifest,
        "all_passed": all(check["passed"] for check in checks.values()),
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(checks)} mechanism checks and {len(raw.manifest)} raw CSVs to {OUTPUT}")
    print(json.dumps({topic: check["results"] for topic, check in checks.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
