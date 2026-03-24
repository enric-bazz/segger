from __future__ import annotations

from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_lsf_segger_benchmark.sh"
DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "benchmark_status_dashboard.sh"
AGG_DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "benchmark_status_dashboard_lsf_multi.sh"
AGG_VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "build_benchmark_validation_table_lsf_multi.sh"


def _run_script(script: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env.update(env)
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=REPO_ROOT,
        env=merged_env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_run_lsf_script_dry_run_creates_compatible_dataset_root(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "1",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    plan_file = dataset_root / "job_plan.tsv"
    gpu0_file = dataset_root / "summaries" / "gpu0.tsv"
    gpu1_file = dataset_root / "summaries" / "gpu1.tsv"
    all_jobs_file = dataset_root / "summaries" / "all_jobs.tsv"
    status_snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    baseline_log = dataset_root / "logs" / "baseline.gpu0.log"
    baseline_script = dataset_root / "bsub" / "baseline.attempt1.sh"
    baseline_export_script = dataset_root / "bsub" / "baseline.export.attempt1.sh"
    align_script = dataset_root / "bsub" / "align_0p01.attempt1.sh"

    assert plan_file.exists()
    assert gpu0_file.exists()
    assert gpu1_file.exists()
    assert all_jobs_file.exists()
    assert status_snapshot.exists()
    assert baseline_log.exists()
    assert baseline_script.exists()
    assert baseline_export_script.exists()
    assert align_script.exists()

    lines = plan_file.read_text().strip().splitlines()
    assert lines[0] == (
        "job\tgroup\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\t"
        "mode\tfragment_mode\tqueue\tgmem_gb\tmem_gb\twall_time"
    )
    assert len(lines) == 12
    assert any(line.startswith("align_0p01\t") for line in lines[1:])
    assert any(line.startswith("align_0p03\t") for line in lines[1:])
    assert any(line.startswith("align_0p10\t") for line in lines[1:])

    status_header = gpu0_file.read_text().splitlines()[0].split("\t")
    assert status_header[-5:] == [
        "segment_status",
        "export_status",
        "segment_job_id",
        "export_job_id",
        "export_note",
    ]

    assert "segger export" not in baseline_script.read_text()
    assert '#BSUB -w "done(segger_xenium_crc_baseline_' in baseline_export_script.read_text()
    assert "--alignment-loss" in align_script.read_text()
    assert "--tissue-type colon" in align_script.read_text()
    assert "--scrna-celltype-column" not in align_script.read_text()

    _run_script(DASHBOARD_SCRIPT, {"ROOT": str(dataset_root)}, "--root", str(dataset_root))


def test_run_lsf_script_uses_split_export_scripts_and_schedulable_fragment_resources(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    fragon_script = dataset_root / "bsub" / "pred_sf1p2_fragon.attempt1.sh"
    export_script = dataset_root / "bsub" / "baseline.export.attempt1.sh"
    gpu0_file = dataset_root / "summaries" / "gpu0.tsv"

    fragon_text = fragon_script.read_text()
    export_text = export_script.read_text()
    gpu0_text = gpu0_file.read_text()

    assert '#BSUB -gpu "num=1:j_exclusive=yes:gmem=60G"' in fragon_text
    assert '#BSUB -R "rusage[mem=256G]"' in fragon_text
    assert '#BSUB -W 12:00' in fragon_text
    assert '#BSUB -w "done(segger_xenium_crc_baseline_' in export_text
    assert "export_skipped_xenium_missing_source" in export_text
    assert "xenium_missing_source_will_skip" in gpu0_text


def test_run_lsf_script_routes_crc_to_gpu_when_eligible_by_default(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
        "ROUTE_ELIGIBLE_TO_GPU": "1",
    }

    _run_script(RUN_SCRIPT, env)

    baseline_script = output_root / "datasets" / "xenium_crc" / "bsub" / "baseline.attempt1.sh"
    assert baseline_script.exists()
    baseline_text = baseline_script.read_text()
    assert "#BSUB -q gpu" in baseline_text
    assert '#BSUB -gpu "num=1:j_exclusive=yes:gmem=30G"' in baseline_text


def test_run_lsf_script_can_force_crc_to_gpu_pro(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
        "ROUTE_ELIGIBLE_TO_GPU": "1",
        "FORCE_GPU_PRO_DATASETS": "xenium_crc",
    }

    _run_script(RUN_SCRIPT, env)

    baseline_script = output_root / "datasets" / "xenium_crc" / "bsub" / "baseline.attempt1.sh"
    assert baseline_script.exists()
    baseline_text = baseline_script.read_text()
    assert "#BSUB -q gpu-pro" in baseline_text
    assert '#BSUB -gpu "num=1:j_exclusive=yes:gmem=30G"' in baseline_text


def test_run_lsf_script_retry_fallback_bumps_gmem_and_escalates_queue(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_breast",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
        "ROUTE_ELIGIBLE_TO_GPU": "1",
        "GPU_QUEUE_MAX_MEM_GB": "384",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_breast"
    baseline_attempt1_err = dataset_root / "logs" / "baseline.attempt1.err"
    baseline_attempt1_err.write_text("CUDA error: out of memory\n")

    _run_script(RUN_SCRIPT, env)

    baseline_attempt2_script = dataset_root / "bsub" / "baseline.attempt2.sh"
    assert baseline_attempt2_script.exists()
    baseline_attempt2_text = baseline_attempt2_script.read_text()
    assert "#BSUB -q gpu-pro" in baseline_attempt2_text
    assert '#BSUB -gpu "num=1:j_exclusive=yes:gmem=50G"' in baseline_attempt2_text
    assert '#BSUB -R "rusage[mem=512G]"' in baseline_attempt2_text

    gpu0_rows = (dataset_root / "summaries" / "gpu0.tsv").read_text()
    assert "gmem_bump=+20G" in gpu0_rows


def test_run_lsf_script_export_only_mode_skips_jobs_without_primary_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    dataset_root = output_root / "datasets" / "xenium_crc"
    baseline_seg_dir = dataset_root / "runs" / "baseline"
    baseline_seg_dir.mkdir(parents=True, exist_ok=True)
    (baseline_seg_dir / "segger_segmentation.parquet").write_text("seg")
    (baseline_seg_dir / ".segment_status").write_text("segment_status=segment_ok\nsegment_rc=0\n")

    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
        "RUN_PRIMARY_JOBS": "0",
        "RUN_EXPORT_JOBS": "1",
        "RUN_ANNDATA_EXPORT": "1",
        "RUN_XENIUM_EXPORT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    baseline_export_script = dataset_root / "bsub" / "baseline.export.attempt1.sh"
    use3d_primary_script = dataset_root / "bsub" / "use3d_true.attempt1.sh"
    assert baseline_export_script.exists()
    assert not use3d_primary_script.exists()

    all_jobs = (dataset_root / "summaries" / "all_jobs.tsv").read_text().splitlines()
    header = all_jobs[0].split("\t")
    rows = {line.split("\t")[0]: dict(zip(header, line.split("\t"))) for line in all_jobs[1:]}
    assert rows["baseline"]["status"] == "pending"
    assert rows["use3d_true"]["status"] == "skipped_filter"
    assert rows["use3d_true"]["export_note"] == "primary_disabled"


def test_run_lsf_script_does_not_reuse_primary_outputs_without_success_status_by_default(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    dataset_root = output_root / "datasets" / "xenium_crc"
    baseline_seg_dir = dataset_root / "runs" / "baseline"
    baseline_export_dir = dataset_root / "exports" / "baseline" / "anndata"
    baseline_seg_dir.mkdir(parents=True, exist_ok=True)
    baseline_export_dir.mkdir(parents=True, exist_ok=True)
    (baseline_seg_dir / "segger_segmentation.parquet").write_text("seg")
    (baseline_export_dir / "segger_segmentation.h5ad").write_text("h5ad")

    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
        "RUN_XENIUM_EXPORT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    all_jobs = (dataset_root / "summaries" / "all_jobs.tsv").read_text().splitlines()
    header = all_jobs[0].split("\t")
    rows = {line.split("\t")[0]: dict(zip(header, line.split("\t"))) for line in all_jobs[1:]}
    assert rows["baseline"]["status"] == "pending"
    assert rows["baseline"]["segment_status"] == "pending"


def test_dashboard_promotes_stale_pending_rows_from_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "1",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"

    baseline_seg = dataset_root / "runs" / "baseline" / "segger_segmentation.parquet"
    baseline_h5ad = dataset_root / "exports" / "baseline" / "anndata" / "segger_segmentation.h5ad"
    use3d_seg = dataset_root / "runs" / "use3d_true" / "segger_segmentation.parquet"
    use3d_h5ad = dataset_root / "exports" / "use3d_true" / "anndata" / "segger_segmentation.h5ad"
    use3d_xenium = dataset_root / "exports" / "use3d_true" / "xenium_explorer" / "seg_experiment.xenium"

    baseline_seg.parent.mkdir(parents=True, exist_ok=True)
    baseline_seg.write_text("seg")
    baseline_h5ad.parent.mkdir(parents=True, exist_ok=True)
    baseline_h5ad.write_text("h5ad")

    use3d_seg.parent.mkdir(parents=True, exist_ok=True)
    use3d_seg.write_text("seg")
    use3d_h5ad.parent.mkdir(parents=True, exist_ok=True)
    use3d_h5ad.write_text("h5ad")
    use3d_xenium.parent.mkdir(parents=True, exist_ok=True)
    use3d_xenium.write_text("xenium")

    _run_script(DASHBOARD_SCRIPT, {"ROOT": str(dataset_root)}, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "ok"
    assert rows["baseline"]["state"] == "done"
    assert rows["use3d_true"]["status"] == "ok"
    assert rows["use3d_true"]["state"] == "done"


def test_dashboard_uses_lsf_job_state_for_transient_rows(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    gpu0_file = dataset_root / "summaries" / "gpu0.tsv"
    gpu1_file = dataset_root / "summaries" / "gpu1.tsv"

    gpu0_file.write_text(
        "\n".join(
            [
                "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file",
                f"baseline\t0\tpending\t0\tqueue=gpu-pro gmem=30G attempt=1 lsf_job=101\t{dataset_root / 'runs' / 'baseline'}\t{dataset_root / 'logs' / 'baseline.gpu0.log'}",
                f"pred_sf1p2_fragoff\t0\tpending\t0\tqueue=gpu-pro gmem=30G attempt=1 lsf_job=202\t{dataset_root / 'runs' / 'pred_sf1p2_fragoff'}\t{dataset_root / 'logs' / 'pred_sf1p2_fragoff.gpu0.log'}",
                "",
            ]
        )
    )
    gpu1_file.write_text(
        "\n".join(
            [
                "job\tgpu\tstatus\telapsed_s\tnote\tseg_dir\tlog_file",
                f"pred_sf1p2_fragon\t1\tpending\t0\tqueue=gpu-pro gmem=60G attempt=1 lsf_job=303\t{dataset_root / 'runs' / 'pred_sf1p2_fragon'}\t{dataset_root / 'logs' / 'pred_sf1p2_fragon.gpu1.log'}",
                "",
            ]
        )
    )

    fake_bin_dir = tmp_path / "fake-bin"
    fake_bin_dir.mkdir()
    fake_bjobs = fake_bin_dir / "bjobs"
    fake_bjobs.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  case \"$arg\" in\n"
        "    101) printf '101 RUN\\n' ;;\n"
        "    202) printf '202 PEND\\n' ;;\n"
        "    303) printf '303 EXIT\\n' ;;\n"
        "  esac\n"
        "done\n"
    )
    os.chmod(fake_bjobs, 0o755)

    dashboard_env = {
        "PATH": f"{fake_bin_dir}:{os.environ.get('PATH', '')}",
        "LSF_SUBMIT_HOST": "local",
    }
    _run_script(DASHBOARD_SCRIPT, dashboard_env, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "running"
    assert rows["baseline"]["state"] == "running"
    assert rows["pred_sf1p2_fragoff"]["status"] == "pending"
    assert rows["pred_sf1p2_fragoff"]["state"] == "pending"
    assert rows["pred_sf1p2_fragon"]["status"] == "lsf_exit"
    assert rows["pred_sf1p2_fragon"]["state"] == "failed"


def test_dashboard_minimal_mode_shows_compact_jobs_overview(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    result = _run_script(DASHBOARD_SCRIPT, {"ROOT": str(dataset_root)}, "--root", str(dataset_root), "--minimal")

    assert "Done Jobs:" in result.stdout
    assert "Failed Jobs:" in result.stdout
    assert "anndata" in result.stdout
    assert "Jobs Overview:" not in result.stdout
    assert "All Jobs Overview:" not in result.stdout
    assert "State Counts:" not in result.stdout


def test_dashboard_marks_export_failures_failed_and_points_to_attempt_err(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    seg_dir = dataset_root / "runs" / "baseline"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segger_segmentation.parquet").write_text("seg")
    (seg_dir / ".export_status").write_text("export_rc=2\n")

    attempt_err = dataset_root / "logs" / "baseline.attempt1.err"
    attempt_err.write_text("export failed\n")

    _run_script(DASHBOARD_SCRIPT, {"LSF_SUBMIT_HOST": "local"}, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "export_error"
    assert rows["baseline"]["state"] == "failed"
    assert rows["baseline"]["log_file"] == str(attempt_err)


def test_dashboard_marks_segmentation_only_rows_done_and_uses_canonical_log(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    seg_dir = dataset_root / "runs" / "baseline"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segger_segmentation.parquet").write_text("seg")

    attempt_err = dataset_root / "logs" / "baseline.attempt1.err"
    attempt_err.write_text("xenium export missing\n")

    _run_script(DASHBOARD_SCRIPT, {"LSF_SUBMIT_HOST": "local"}, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "segment_only"
    assert rows["baseline"]["state"] == "done"
    assert rows["baseline"]["log_file"] == str(dataset_root / "logs" / "baseline.gpu0.log")


def test_dashboard_treats_skipped_xenium_export_as_success(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    seg_dir = dataset_root / "runs" / "baseline"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segger_segmentation.parquet").write_text("seg")
    (seg_dir / ".segment_status").write_text("segment_status=segment_ok\nsegment_rc=0\n")
    (seg_dir / ".export_status").write_text(
        "export_status=export_skipped_xenium_missing_source\n"
        "export_rc=0\n"
        "export_note=xenium_missing_source\n"
    )
    anndata = dataset_root / "exports" / "baseline" / "anndata" / "segger_segmentation.h5ad"
    anndata.parent.mkdir(parents=True, exist_ok=True)
    anndata.write_text("h5ad")

    _run_script(DASHBOARD_SCRIPT, {"LSF_SUBMIT_HOST": "local"}, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "ok"
    assert rows["baseline"]["state"] == "done"
    assert rows["baseline"]["export_status"] == "export_skipped_xenium_missing_source"


def test_dashboard_uses_segment_stage_status_for_oom_failures(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "0",
    }

    _run_script(RUN_SCRIPT, env)

    dataset_root = output_root / "datasets" / "xenium_crc"
    seg_dir = dataset_root / "runs" / "baseline"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / ".segment_status").write_text("segment_status=segment_oom\nsegment_rc=140\n")

    _run_script(DASHBOARD_SCRIPT, {"LSF_SUBMIT_HOST": "local"}, "--root", str(dataset_root))

    snapshot = dataset_root / "summaries" / "status_snapshot.tsv"
    rows = {}
    with snapshot.open() as handle:
        header = handle.readline().strip().split("\t")
        for line in handle:
            values = line.strip().split("\t")
            rows[values[0]] = dict(zip(header, values))

    assert rows["baseline"]["status"] == "segment_oom"
    assert rows["baseline"]["state"] == "failed"


def test_aggregate_wrappers_merge_status_and_validation_tables(tmp_path: Path) -> None:
    output_root = tmp_path / "lsf_benchmark_multi"
    env = {
        "OUTPUT_ROOT": str(output_root),
        "DATASET_KEYS": "xenium_crc,xenium_nsclc",
        "DRY_RUN": "1",
        "AUTO_SUBMIT": "0",
        "RUN_VALIDATION_TABLE": "0",
        "RUN_STATUS_SNAPSHOT": "1",
    }

    _run_script(RUN_SCRIPT, env)

    for dataset in ("xenium_crc", "xenium_nsclc"):
        validation_tsv = output_root / "datasets" / dataset / "summaries" / "validation_metrics.tsv"
        validation_tsv.write_text("job\tvalidate_status\nbaseline\tok\n")

    dashboard_result = _run_script(AGG_DASHBOARD_SCRIPT, {"OUTPUT_ROOT": str(output_root)}, "--root", str(output_root))
    _run_script(AGG_VALIDATION_SCRIPT, {"OUTPUT_ROOT": str(output_root)}, "--root", str(output_root), "--no-refresh")

    aggregate_status = output_root / "summaries" / "aggregate_status_snapshot.tsv"
    aggregate_validation = output_root / "summaries" / "aggregate_validation_metrics.tsv"

    assert aggregate_status.exists()
    assert aggregate_validation.exists()
    assert "Aggregate Benchmark Dashboard" in dashboard_result.stdout
    assert "Dataset Summary:" in dashboard_result.stdout
    assert "All Jobs:" in dashboard_result.stdout

    validation_lines = aggregate_validation.read_text().strip().splitlines()
    assert validation_lines[0] == "dataset\tjob\tvalidate_status"
    assert any(line.startswith("xenium_crc\tbaseline\tok") for line in validation_lines[1:])
    assert any(line.startswith("xenium_nsclc\tbaseline\tok") for line in validation_lines[1:])

    aggregate_status_lines = aggregate_status.read_text().strip().splitlines()
    assert aggregate_status_lines[0].startswith("dataset\tjob\t")
