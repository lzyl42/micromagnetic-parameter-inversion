"""Tests for the prepare pipeline, npz schema, split and contracts (offline).

Synthetic raw layouts all live under tmp_path (``MICROMAG_DATA_ROOT`` redirected) with no
dependence on data/raw; no simulations/training are run.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from micromagnetic_parameter_inversion import paths, training_data
from micromagnetic_parameter_inversion.training_config import (
    SplitConfig,
    SplitMinCounts,
    SplitRatios,
)
from micromagnetic_parameter_inversion.training_data import (
    DataError,
    InputContract,
    load_protocol_snapshot,
    make_split,
)

_PSIDS = ("grp_a", "grp_b", "grp_c", "grp_d")
_PULSES = ("p0", "p1")


def _write_raw_dataset(
    root: Path,
    dataset: str,
    psids: Sequence[str] = _PSIDS,
    pulses: Sequence[str] = _PULSES,
    n_steps: int = 8,
    sample_interval_s: float = 1.0e-9,
    per_psid_interval: Mapping[str, float] | None = None,
    numerics: bool = False,
) -> None:
    """Synthesize a minimal raw layout: config.yaml + index.csv + runs/<pulse>/trajectory.csv.

    Numerics is absent (older snapshots are valid); each pulse has a different
    m_x pattern so the pulse dimension order of x can be checked; t_s uses repr
    to guarantee exact float64 roundtrip.
    """
    for i, psid in enumerate(psids):
        psid_dir = root / "raw" / dataset / psid
        (psid_dir / "runs").mkdir(parents=True, exist_ok=True)
        alpha = 0.005 * (i + 1)
        ku = 1.0e4 * (i + 1)
        interval = (per_psid_interval or {}).get(psid, sample_interval_s)
        config: dict[str, Any] = {
            "dataset_name": dataset,
            "recording": {"sample_interval_s": interval, "sample_count": n_steps},
            "pulses": [
                {
                    "pulse_id": pulse,
                    "b_ext_amplitude_mT": 10.0 + j,
                    "direction": [1.0, 0.0, 0.0],
                    "duration_s": 1.0e-9,
                }
                for j, pulse in enumerate(pulses)
            ],
        }
        if numerics:
            config["numerics"] = {"edge_smooth": 1, "solver": 5}
        (psid_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        rows = [
            ",".join(
                (
                    psid,
                    pulse,
                    repr(alpha),
                    repr(ku),
                    "0.01",
                    "0",
                    "0",
                    "1e-09",
                    f"runs/{pulse}/trajectory.csv",
                )
            )
            for pulse in pulses
        ]
        (psid_dir / "index.csv").write_text(
            ",".join(training_data.INDEX_COLUMNS) + "\n" + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        for j, pulse in enumerate(pulses):
            lines = ["sample_index,t_s,m_x,m_y,m_z"]
            for k in range(n_steps):
                lines.append(f"{k},{k * interval!r},{0.5 * j + 0.01 * k:.6g},0.2,0.9")
            pulse_dir = psid_dir / "runs" / pulse
            pulse_dir.mkdir(exist_ok=True)
            (pulse_dir / "trajectory.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_config_text(
    dataset: str,
    *,
    pulse_order: Sequence[str] | None = None,
    ratios: tuple[float, float, float] = (0.5, 0.25, 0.25),
    min_per_split: tuple[int, int, int] = (2, 1, 1),
) -> str:
    order = "null" if pulse_order is None else "[" + ", ".join(pulse_order) + "]"
    return (
        f"dataset_name: {dataset}\n"
        "run_name: run_001\n"
        f"data:\n  pulse_order: {order}\n"
        "training: {seed: 7, device: cpu}\n"
        f"split:\n  seed: 7\n"
        f"  ratios: {{train: {ratios[0]}, val: {ratios[1]}, test: {ratios[2]}}}\n"
        f"  min_per_split: {{train: {min_per_split[0]}, val: {min_per_split[1]},"
        f" test: {min_per_split[2]}}}\n"
        "output_dir: null\n"
    )


def _load_script() -> Any:
    """Load the prepare script module by path (scripts/ is not a package)."""
    script_path = paths.PROJECT_ROOT / "scripts" / "prepare_training_samples.py"
    spec = importlib.util.spec_from_file_location("prepare_training_samples", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect MICROMAG_DATA_ROOT to a temporary directory."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(root))
    return root


def _run_prepare(
    data_root: Path, dataset: str, *, config_kwargs: dict[str, Any] | None = None, ids: str = "all"
) -> int:
    config_kwargs = config_kwargs or {}
    config_path = data_root / "training_cfg.yaml"
    config_path.write_text(_training_config_text(dataset, **config_kwargs), encoding="utf-8")
    return _load_script().main(["--config", str(config_path), "--parameter-set-ids", *ids.split()])


def test_prepare_all_end_to_end(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = "ds_e2e"
    _write_raw_dataset(data_root, dataset, per_psid_interval={"grp_a": 2.0e-9})
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    for psid in _PSIDS:
        assert (samples_dir / f"{psid}.npz").is_file()
    assert (samples_dir / "dataset_meta.yaml").is_file()
    assert (samples_dir / "split.yaml").is_file()

    sample = training_data.load_sample(samples_dir, "grp_a")
    assert sample.x.shape == (2, 8, 3) and sample.x.dtype == np.float32
    assert sample.y.dtype == np.float32 and sample.y.shape == (2,)
    assert sample.t_s.dtype == np.float64 and sample.t_s.shape == (8,)
    assert sample.pulse_ids == ("p0", "p1")
    expected_t = np.array(
        [k * 2.0e-9 for k in range(8)]
    )  # first snapshot = lowest psid in ascending order
    np.testing.assert_array_equal(sample.t_s, expected_t)
    expected_m = np.loadtxt(
        data_root / "raw" / dataset / "grp_a" / "runs" / "p0" / "trajectory.csv",
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 4),
        ndmin=2,
    ).astype(np.float32)
    np.testing.assert_array_equal(sample.x[0], expected_m)

    meta = training_data.load_dataset_meta(samples_dir)
    assert meta.dataset_name == dataset
    assert meta.members == tuple(sorted(_PSIDS))
    assert meta.pulse_order == ("p0", "p1")
    assert meta.n_time_steps == 8
    assert meta.labels["grp_a"] == (0.005, 1.0e4)
    assert meta.protocol is not None
    assert (
        meta.protocol["sample_interval_s"] == 2.0e-9
    )  # snapshot of the first psid in ascending order
    assert meta.protocol["pulses"]["p1"]["b_ext_amplitude_t"] == pytest.approx(11.0e-3)
    assert set(meta.config_sha256 or {}) == set(meta.members)
    assert meta.source_index_relpaths is not None
    assert meta.source_index_relpaths["grp_b"] == "grp_b/index.csv"
    assert meta.generated_at

    split = training_data.load_split(samples_dir, meta)
    assert (len(split.train), len(split.val), len(split.test)) == (2, 1, 1)
    all_members = split.train + split.val + split.test
    assert sorted(all_members) == list(meta.members)
    assert len(set(all_members)) == len(all_members)
    assert "prepare complete" in capsys.readouterr().out


def test_prepare_explicit_psids_subset(data_root: Path) -> None:
    dataset = "ds_subset"
    _write_raw_dataset(data_root, dataset, psids=("grp_a", "grp_b", "grp_z"))
    kwargs = {"ratios": (0.5, 0.5, 0.0), "min_per_split": (1, 1, 0)}
    assert _run_prepare(data_root, dataset, config_kwargs=kwargs, ids="grp_b grp_z") == 0
    meta = training_data.load_dataset_meta(data_root / "samples" / dataset)
    assert meta.members == ("grp_b", "grp_z")  # unselected grp_a does not enter the manifest


def test_prepare_rejects_existing_outputs(
    data_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = "ds_nooverwrite"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    npz_bytes = (data_root / "samples" / dataset / "grp_a.npz").read_bytes()
    assert _run_prepare(data_root, dataset) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (data_root / "samples" / dataset / "grp_a.npz").read_bytes() == npz_bytes


def test_prepare_rejects_min_violation_before_write(data_root: Path) -> None:
    dataset = "ds_min"
    _write_raw_dataset(data_root, dataset)
    exit_code = _run_prepare(data_root, dataset, config_kwargs={"min_per_split": (2, 1, 5)})
    assert exit_code == 2
    assert not (data_root / "samples" / dataset).exists()  # refused before writing samples


def test_prepare_configured_pulse_order(data_root: Path) -> None:
    dataset = "ds_order"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset, config_kwargs={"pulse_order": ["p1", "p0"]}) == 0
    samples_dir = data_root / "samples" / dataset
    sample = training_data.load_sample(samples_dir, "grp_c")
    assert sample.pulse_ids == ("p1", "p0")
    expected_m = np.loadtxt(
        data_root / "raw" / dataset / "grp_c" / "runs" / "p1" / "trajectory.csv",
        delimiter=",",
        skiprows=1,
        usecols=(2, 3, 4),
        ndmin=2,
    ).astype(np.float32)
    np.testing.assert_array_equal(
        sample.x[0], expected_m
    )  # the P dimension of x follows the configured order


def test_prepare_invalid_pulse_order_rejected(
    data_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = "ds_badorder"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset, config_kwargs={"pulse_order": ["p0", "p9"]}) == 2
    assert _run_prepare(data_root, dataset, config_kwargs={"pulse_order": ["p0", "p0"]}) == 2
    assert "pulse" in capsys.readouterr().err


def test_prepare_t1_single_row(data_root: Path) -> None:
    dataset = "ds_t1"
    _write_raw_dataset(data_root, dataset, psids=("grp_a", "grp_b"), n_steps=1)
    kwargs = {"ratios": (0.5, 0.5, 0.0), "min_per_split": (1, 1, 0)}
    assert _run_prepare(data_root, dataset, config_kwargs=kwargs) == 0
    sample = training_data.load_sample(data_root / "samples" / dataset, "grp_a")
    assert sample.x.shape == (2, 1, 3)  # ndmin=2 prevents a single row from being squeezed
    assert sample.t_s.shape == (1,)


def test_protocol_snapshot_schema_and_mt_conversion(data_root: Path) -> None:
    _write_raw_dataset(
        data_root, "ds_proto", numerics=True
    )  # with numerics present, still only two sections are read
    protocol = load_protocol_snapshot(data_root / "raw" / "ds_proto", _PSIDS, None)
    assert protocol.pulse_order == ("p0", "p1")
    assert protocol.n_time_steps == 8
    assert protocol.sample_interval_s == 1.0e-9
    assert protocol.pulses["p0"]["b_ext_amplitude_t"] == pytest.approx(0.01)  # 10 mT → T
    assert protocol.pulses["p1"]["b_ext_amplitude_t"] == pytest.approx(0.011)
    # explicit order = duplicate-free permutation of the snapshot set
    reversed_protocol = load_protocol_snapshot(data_root / "raw" / "ds_proto", _PSIDS, ("p1", "p0"))
    assert reversed_protocol.pulse_order == ("p1", "p0")
    with pytest.raises(DataError):
        load_protocol_snapshot(data_root / "raw" / "ds_proto", _PSIDS, ("p0", "p9"))


def test_make_split_largest_remainder_and_tie() -> None:
    psids = [f"p{i}" for i in range(6)]
    config = SplitConfig(
        seed=7,
        ratios=SplitRatios(0.5, 0.25, 0.25),
        min_per_split=SplitMinCounts(1, 1, 1),
    )
    split = make_split(psids, config)
    assert (len(split.train), len(split.val), len(split.test)) == (3, 2, 1)  # tie: val before test
    again = make_split(psids, config)
    assert (again.train, again.val, again.test) == (split.train, split.val, split.test)
    all_members = split.train + split.val + split.test
    assert sorted(all_members) == sorted(psids)
    with pytest.raises(DataError, match="insufficient samples"):
        make_split(
            psids,
            SplitConfig(
                seed=7,
                ratios=SplitRatios(0.5, 0.25, 0.25),
                min_per_split=SplitMinCounts(2, 2, 5),
            ),
        )


def test_load_sample_contract_checks(data_root: Path) -> None:
    dataset = "ds_contract"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    reference = training_data.load_sample(samples_dir, "grp_a")
    contract = InputContract(pulse_order=reference.pulse_ids, n_time_steps=8, t_s=reference.t_s)
    assert training_data.load_sample(samples_dir, "grp_a", contract).x.shape == (2, 8, 3)
    wrong_t = InputContract(pulse_order=reference.pulse_ids, n_time_steps=7, t_s=reference.t_s[:-1])
    with pytest.raises(DataError, match="contract"):
        training_data.load_sample(samples_dir, "grp_a", wrong_t)
    wrong_order = InputContract(pulse_order=("p1", "p0"), n_time_steps=8, t_s=reference.t_s)
    with pytest.raises(DataError, match="pulse order"):
        training_data.load_sample(samples_dir, "grp_a", wrong_order)
    shifted = reference.t_s.copy()
    shifted[0] += 1e-12  # far below any loose tolerance: the bitwise-equality contract must reject
    wrong_time = InputContract(pulse_order=reference.pulse_ids, n_time_steps=8, t_s=shifted)
    with pytest.raises(DataError, match="t_s"):
        training_data.load_sample(samples_dir, "grp_a", wrong_time)
    with pytest.raises(DataError, match="does not exist"):
        training_data.load_sample(samples_dir, "missing_psid")


def test_npz_schema_dtypes(data_root: Path) -> None:
    dataset = "ds_dtypes"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    with np.load(data_root / "samples" / dataset / "grp_a.npz", allow_pickle=False) as archive:
        assert archive["x"].dtype == np.dtype("float32") and archive["x"].shape == (2, 8, 3)
        assert archive["y"].dtype == np.dtype("float32") and archive["y"].shape == (2,)
        assert archive["t_s"].dtype == np.dtype("float64") and archive["t_s"].shape == (8,)
        assert archive["parameter_set_id"].dtype == np.dtype("<U16")
        assert str(archive["parameter_set_id"].item()) == "grp_a"
        assert archive["pulse_ids"].dtype.kind == "U"
        assert archive["pulse_ids"].tolist() == ["p0", "p1"]
        assert sorted(archive.files) == ["parameter_set_id", "pulse_ids", "t_s", "x", "y"]


def test_split_yaml_tampering_rejected(data_root: Path) -> None:
    dataset = "ds_tamper"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    meta = training_data.load_dataset_meta(samples_dir)
    split_path = samples_dir / "split.yaml"

    split_path.write_text(
        "seed: 7\n"
        "ratios: {train: 0.5, val: 0.25, test: 0.25}\n"
        "train: [grp_a, grp_b]\nval: [grp_b]\ntest: [grp_c]\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="not disjoint"):
        training_data.load_split(samples_dir, meta)

    split_path.write_text(
        "seed: 7\n"
        "ratios: {train: 0.5, val: 0.25, test: 0.25}\n"
        "train: [grp_a, grp_b]\nval: [grp_c]\ntest: [grp_d]\n"
        "extra_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="unknown fields"):
        training_data.load_split(samples_dir, meta)

    split_path.write_text(
        "seed: 7\n"
        "ratios: {train: 0.5, val: 0.25, test: 0.25}\n"
        "train: [grp_a, grp_b]\nval: [grp_c]\ntest: [grp_d]\n",
        encoding="utf-8",
    )
    (samples_dir / "grp_a.npz").unlink()
    with pytest.raises(DataError, match="does not exist"):
        training_data.load_split(samples_dir, meta)


def test_stale_npz_not_scanned(data_root: Path) -> None:
    dataset = "ds_stale"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    np.savez_compressed(
        samples_dir / "deadbeef.npz", x=np.zeros((1, 1, 3), dtype=np.float32)
    )  # stale npz left in the directory: not in the manifest, not scanned
    meta = training_data.load_dataset_meta(samples_dir)
    assert "deadbeef" not in meta.members
    split = training_data.load_split(samples_dir, meta)
    datasets = training_data.build_datasets(samples_dir, meta, split)
    assert [len(ds) for ds in datasets] == [len(split.train), len(split.val), len(split.test)]


def test_prepare_raw_contract_violations(
    data_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = "ds_badraw"
    _write_raw_dataset(data_root, dataset)
    (data_root / "raw" / dataset / "grp_b" / "runs" / "p1" / "trajectory.csv").unlink()
    assert _run_prepare(data_root, dataset) == 2
    assert "trajectory.csv" in capsys.readouterr().err

    index_path = data_root / "raw" / dataset / "grp_b" / "index.csv"
    index_path.write_text("pulse_id,alpha\np0,0.1\n", encoding="utf-8")  # missing fixed columns
    assert _run_prepare(data_root, dataset) == 2
    assert "missing fixed columns" in capsys.readouterr().err

    _write_raw_dataset(data_root, dataset)  # restore, then inject an extra pulse row
    index_path = data_root / "raw" / dataset / "grp_b" / "index.csv"
    index_path.write_text(
        index_path.read_text(encoding="utf-8")
        + f"grp_b,p9,0.01,{1.0e4!r},0.01,0,0,1e-09,runs/p0/trajectory.csv\n",
        encoding="utf-8",
    )
    assert _run_prepare(data_root, dataset) == 2
    assert "extra" in capsys.readouterr().err

    _write_raw_dataset(data_root, dataset)  # restore, then tamper with row 2's alpha
    index_path = data_root / "raw" / dataset / "grp_b" / "index.csv"
    lines = index_path.read_text(encoding="utf-8").splitlines()
    fields = lines[2].split(",")
    assert fields[0] == "grp_b" and fields[1] == "p1"
    fields[2] = "0.02"  # alpha does not match the first row of the group
    lines[2] = ",".join(fields)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _run_prepare(data_root, dataset) == 2
    assert "mismatch" in capsys.readouterr().err


def test_prepare_placeholder_and_mixed_all(
    data_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = data_root / "placeholder.yaml"
    config_path.write_text(
        "dataset_name: PLACEHOLDER_DATASET_NAME\nrun_name: r\n", encoding="utf-8"
    )
    assert _load_script().main(["--config", str(config_path), "--parameter-set-ids", "all"]) == 2
    assert "placeholder" in capsys.readouterr().err

    dataset = "ds_mixed"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset, ids="all grp_a") == 2
    assert "mixed" in capsys.readouterr().err


def test_load_split_allows_empty_groups_rejects_duplicates(data_root: Path) -> None:
    dataset = "ds_empty_split"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    meta = training_data.load_dataset_meta(samples_dir)
    psids = sorted(meta.members)

    def _write_split(train: list[str], val: list[str], test: list[str]) -> None:
        (samples_dir / "split.yaml").write_text(
            yaml.safe_dump(
                {
                    "seed": 5,
                    "ratios": {"train": 0.5, "val": 0.25, "test": 0.25},
                    "train": train,
                    "val": val,
                    "test": test,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    _write_split(psids[:4], psids[4:], [])  # empty test group is legal (min_per_split allows 0)
    split = training_data.load_split(samples_dir, meta)
    assert split.test == ()
    assert split.members_of("test") == ()
    assert split.train == tuple(psids[:4])
    _write_split([], psids[:2], psids[2:])  # an empty train is likewise accepted by load_split
    split = training_data.load_split(samples_dir, meta)
    assert split.train == ()
    _write_split(
        psids[:2] + [psids[0]], psids[2:4], psids[4:]
    )  # duplicates within a group are still rejected
    with pytest.raises(training_data.DataError, match="duplicate"):
        training_data.load_split(samples_dir, meta)


def test_load_dataset_meta_meta_path_equivalence_and_bad_schema(data_root: Path) -> None:
    dataset = "ds_meta_path"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    default = training_data.load_dataset_meta(samples_dir)
    explicit = training_data.load_dataset_meta(
        samples_dir, meta_path=samples_dir / "dataset_meta.yaml"
    )
    assert explicit == default  # default/explicit paths parse the same schema with identical fields

    broken = samples_dir / "broken_meta.yaml"
    broken.write_text(yaml.safe_dump({"dataset_name": dataset, "surprise": 1}), encoding="utf-8")
    with pytest.raises(training_data.DataError, match="unknown fields"):
        training_data.load_dataset_meta(samples_dir, meta_path=broken)
    meta_file = samples_dir / "dataset_meta.yaml"
    original = meta_file.read_text(encoding="utf-8")
    meta_file.write_text(original + "surprise: 1\n", encoding="utf-8")
    with pytest.raises(training_data.DataError, match="unknown fields"):
        training_data.load_dataset_meta(
            samples_dir
        )  # bad schema at the default path is rejected too


def test_dataset_loads_each_npz_once_and_skips_test(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train/val read each npz from disk exactly once (cache reuse + contract never
    re-reads); test is not read.
    """
    dataset = "ds_read_once"
    _write_raw_dataset(data_root, dataset)
    assert _run_prepare(data_root, dataset) == 0
    samples_dir = data_root / "samples" / dataset
    meta = training_data.load_dataset_meta(samples_dir)
    split = training_data.load_split(samples_dir, meta)

    loaded: list[Path] = []
    original_load = np.load

    def counting_load(path: object, *args: object, **kwargs: object) -> object:
        loaded.append(Path(str(path)))
        return original_load(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(np, "load", counting_load)
    train_set = training_data.TrajectoryDataset(samples_dir, meta, split.train)
    first = train_set.samples[0]  # public attribute: contract frozen from the first cached sample
    contract = training_data.InputContract(
        pulse_order=first.pulse_ids, n_time_steps=int(first.x.shape[1]), t_s=first.t_s
    )
    train_set.validate_contract(contract)  # pure in-memory validation, no disk re-read
    training_data.TrajectoryDataset(samples_dir, meta, split.val, contract=contract)
    expected = sorted(samples_dir / f"{psid}.npz" for psid in (*split.train, *split.val))
    assert sorted(loaded) == expected
    assert len(loaded) == len(set(loaded))  # each file exactly once
    test_paths = {samples_dir / f"{psid}.npz" for psid in split.test}
    assert not (set(loaded) & test_paths)  # test is not read


def test_resolve_psids_accepts_all_tuple(data_root: Path) -> None:
    """--parameter-set-ids Sequence compatibility: ("all",) and ["all"] are equivalent."""
    dataset = "ds_all_tuple"
    _write_raw_dataset(data_root, dataset)
    script = _load_script()
    raw_dir = data_root / "raw" / dataset
    assert script._resolve_psids(raw_dir, ("all",)) == script._resolve_psids(raw_dir, ["all"])
    with pytest.raises(training_data.DataError, match="mixed"):
        script._resolve_psids(raw_dir, ("all", "grp_a"))
