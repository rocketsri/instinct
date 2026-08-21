from __future__ import annotations

import numpy as np

from instinct.p5_development.generator import CoordinateGenerator


def test_generated_function_commutes_with_neuron_relabeling() -> None:
    generator = CoordinateGenerator()
    rng = np.random.default_rng(5)
    pre = np.linspace(0.1, 0.9, 7)
    post = np.linspace(0.2, 0.8, 5)
    activity = rng.normal(size=7)
    p, q = rng.permutation(7), rng.permutation(5)
    expected = generator.function_output(activity, pre, post)
    got = generator.function_output(activity[p], pre[p], post[q])
    assert np.allclose(got, expected[q])


def test_local_update_is_unaffected_by_remote_neurons() -> None:
    generator = CoordinateGenerator()
    rng = np.random.default_rng(9)
    pre, post = np.linspace(0.1, 0.9, 6), np.linspace(0.2, 0.8, 4)
    x, y, error = rng.normal(size=6), rng.normal(size=4), rng.normal(size=4)
    reference = generator.local_delta(x, y, error, pre, post, step=2)
    changed_x, changed_y, changed_error = rng.normal(size=6), rng.normal(size=4), rng.normal(size=4)
    changed_x[2], changed_y[1], changed_error[1] = x[2], y[1], error[1]
    changed = generator.local_delta(changed_x, changed_y, changed_error, pre, post, step=2)
    assert changed[1, 2] == reference[1, 2]


def test_duplicate_width_preserves_generated_function() -> None:
    generator = CoordinateGenerator()
    pre, post = np.linspace(0.1, 0.9, 5), np.array([0.25, 0.75])
    activity = np.sin(2 * np.pi * pre)
    original = generator.function_output(activity, pre, post)
    duplicated = generator.function_output(np.repeat(activity, 2), np.repeat(pre, 2), post)
    assert np.allclose(duplicated, original, atol=1e-12)


def test_serialized_program_excludes_no_hidden_size_dependent_state() -> None:
    generator = CoordinateGenerator()
    manifest = generator.serialization_manifest()
    assert manifest["total_bytes"] == generator.code_bytes
    assert manifest["coefficient_count"] == 4
    assert all(
        manifest[key] == 0
        for key in (
            "coordinate_embedding_bytes",
            "architecture_token_bytes",
            "task_token_bytes",
            "broadcast_module_bytes",
        )
    )
