#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE_EOF'
Aggregate per-dataset benchmark status snapshots produced by
benchmark_status_dashboard.sh.

Usage:
  bash scripts/benchmark_status_dashboard_lsf_multi.sh [options]

Options:
  --root <dir>          Top-level benchmark root, dataset container root,
                        or a single dataset root
                        (default: /omics/groups/OE0606/internal/elihei/projects/segger_lsf_benchmark_fixed)
  --out-tsv <file>      Output aggregate TSV path
                        (default: <root>/summaries/aggregate_status_snapshot.tsv)
  --no-refresh          Do not rerun the per-dataset dashboard script
  --watch [sec]         Refresh every N seconds (default: 20)
  --recent-limit <n>    Show the last N parsed LSF runs (default: 10, 0 = all)
  --minimal             Minimal terminal output with compact jobs overview
  --no-color            Disable ANSI colors
  -h, --help            Show this help
USAGE_EOF
}

ROOT="/omics/groups/OE0606/internal/elihei/projects/segger_lsf_benchmark_fixed"
OUT_TSV=""
REFRESH=1
WATCH_SEC=0
RECENT_LIMIT=10
MINIMAL_MODE=0
NO_COLOR=0

SCRIPT_DIR=""
DATASETS_DIR=""
declare -a DATASET_DIRS=()
declare -a DASH_WARNINGS=()

C_RESET=""
C_BOLD=""
C_BOLD_OFF=""
C_GREEN=""
C_RED=""
C_YELLOW=""
C_BLUE=""
C_CYAN=""

require_value() {
  local opt="$1"
  if [[ $# -lt 2 ]] || [[ -z "${2}" ]] || [[ "${2}" == -* ]]; then
    echo "ERROR: ${opt} requires a value." >&2
    exit 1
  fi
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

draw_progress_bar() {
  local current="$1"
  local total="$2"
  local width=40
  local fill=0
  local pct=0
  local empty left right

  if [[ "${total}" -gt 0 ]]; then
    fill=$((current * width / total))
    pct=$((current * 100 / total))
  fi
  empty=$((width - fill))
  left="$(printf '%*s' "${fill}" '' | tr ' ' '#')"
  right="$(printf '%*s' "${empty}" '' | tr ' ' '-')"
  printf "[%s%s] %d/%d (%d%%)" "${left}" "${right}" "${current}" "${total}" "${pct}"
}

parquet_row_count_fast() {
  local parquet_path="$1"
  python3 - "${parquet_path}" <<'PY' 2>/dev/null
import sys

path = sys.argv[1]
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    print("")
    raise SystemExit(0)

try:
    meta = pq.ParquetFile(path).metadata
    if meta is None:
        print("")
    else:
        print(str(int(meta.num_rows)))
except Exception:
    print("")
PY
}

add_warning() {
  local msg="$1"
  DASH_WARNINGS+=("${msg}")
}

configure_colors() {
  if [[ "${NO_COLOR}" == "0" ]] && [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    C_BOLD_OFF=$'\033[22m'
    C_GREEN=$'\033[32m'
    C_RED=$'\033[31m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_CYAN=$'\033[36m'
  else
    C_RESET=""
    C_BOLD=""
    C_BOLD_OFF=""
    C_GREEN=""
    C_RED=""
    C_YELLOW=""
    C_BLUE=""
    C_CYAN=""
  fi
}

render_tsv_table() {
  awk -F'\t' '
    function repeat_char(ch, n, out, i) {
      out = ""
      for (i = 0; i < n; i++) out = out ch
      return out
    }
    function visible_len(s, t) {
      t = s
      gsub(/\033\[[0-9;]*m/, "", t)
      return length(t)
    }
    function print_cell(val, width, vis, i) {
      vis = visible_len(val)
      printf(" %s", val)
      for (i = vis; i < width; i++) {
        printf(" ")
      }
      printf(" |")
    }
    {
      rows = NR
      if ($1 == "__ROW_SEP__") {
        row_sep[NR] = 1
        next
      }
      if (NF > ncols) ncols = NF
      for (i = 1; i <= NF; i++) {
        val = $i
        sub(/\r$/, "", val)
        cells[NR, i] = val
        vis = visible_len(val)
        if (vis > widths[i]) widths[i] = vis
      }
    }
    END {
      if (rows == 0) exit

      sep = "+"
      for (i = 1; i <= ncols; i++) {
        sep = sep repeat_char("-", widths[i] + 2) "+"
      }

      print sep
      printf("|")
      for (i = 1; i <= ncols; i++) {
        print_cell(cells[1, i], widths[i])
      }
      printf("\n")
      print sep

      for (r = 2; r <= rows; r++) {
        if (row_sep[r]) {
          print sep
          continue
        }
        printf("|")
        for (i = 1; i <= ncols; i++) {
          print_cell(cells[r, i], widths[i])
        }
        printf("\n")
      }
      print sep
    }
  '
}

colorize_rows_by_state_column() {
  local state_col="$1"
  awk -F'\t' \
    -v state_col="${state_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "partial") return c_partial
        if (lower == "reference") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        state = (state_col > 0 && state_col <= NF) ? $state_col : ""
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

colorize_rows_by_status_column() {
  local status_col="$1"
  awk -F'\t' \
    -v status_col="${status_col}" \
    -v c_done="${C_GREEN}" \
    -v c_run="${C_BLUE}" \
    -v c_fail="${C_RED}" \
    -v c_pending="${C_YELLOW}" \
    -v c_partial="${C_CYAN}" \
    -v c_reset="${C_RESET}" \
    '
      function state_from_status(status, lower) {
        lower = tolower(status)
        if (status == "" || status == "-" || lower == "<none>") return "pending"
        if (lower ~ /running|in_progress/) return "running"
        if (lower == "partial") return "partial"
        if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
        if (lower ~ /^skipped_/ && lower != "skipped_existing") return "pending"
        if (lower == "pending" || lower == "planned") return "pending"
        if (lower == "reference") return "reference"
        return "failed"
      }
      function color_for_state(s, lower) {
        lower = tolower(s)
        if (lower == "done") return c_done
        if (lower == "running") return c_run
        if (lower == "failed") return c_fail
        if (lower == "pending") return c_pending
        if (lower == "partial") return c_partial
        if (lower == "reference") return c_partial
        return ""
      }
      NR == 1 { print; next }
      {
        status = (status_col > 0 && status_col <= NF) ? $status_col : ""
        state = state_from_status(status)
        color = color_for_state(state)
        if (color != "") {
          for (i = 1; i <= NF; i++) {
            $i = color $i c_reset
          }
        }
        print
      }
    ' OFS=$'\t'
}

colorize_rows_single_color() {
  local color="${1:-}"
  awk -F'\t' -v color="${color}" -v c_reset="${C_RESET}" '
    NR == 1 { print; next }
    {
      if (color != "") {
        for (i = 1; i <= NF; i++) {
          $i = color $i c_reset
        }
      }
      print
    }
  ' OFS=$'\t'
}

normalize_root_layout() {
  local trimmed="${ROOT%/}"
  if [[ -z "${trimmed}" ]]; then
    ROOT="/"
    return
  fi
  if [[ "$(basename "${trimmed}")" == "datasets" ]]; then
    ROOT="$(cd "${trimmed}/.." && pwd)"
  else
    ROOT="${trimmed}"
  fi
}

collect_dataset_dirs() {
  local d
  DATASET_DIRS=()
  DATASETS_DIR=""

  if [[ -d "${ROOT}/datasets" ]]; then
    DATASETS_DIR="${ROOT}/datasets"
    for d in "${DATASETS_DIR}"/*; do
      [[ -d "${d}" ]] || continue
      DATASET_DIRS+=("${d}")
    done
    return
  fi

  if [[ -f "${ROOT}/job_plan.tsv" ]]; then
    DATASETS_DIR="$(dirname "${ROOT}")"
    DATASET_DIRS=("${ROOT}")
    return
  fi

  if [[ -d "${ROOT}" ]]; then
    DATASETS_DIR="${ROOT}"
    for d in "${ROOT}"/*; do
      [[ -d "${d}" ]] || continue
      [[ -f "${d}/job_plan.tsv" ]] || continue
      DATASET_DIRS+=("${d}")
    done
  fi
}

refresh_dataset_snapshot() {
  local dataset_dir="$1"
  bash "${SCRIPT_DIR}/benchmark_status_dashboard.sh" --root "${dataset_dir}" >/dev/null 2>&1
}

parse_lsf_elapsed_seconds() {
  local out_log="$1"
  python3 - "${out_log}" <<'PY'
import re
import sys
from datetime import datetime

path = sys.argv[1]
started = None
terminated = None
pat = re.compile(r"^(Started at|Terminated at)\s+(.*)$")

def parse_stamp(text: str):
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    parts = text.split()
    # Some clusters append a timezone token (e.g. CET); try without it.
    if len(parts) >= 6:
        candidates.append(" ".join(parts[:5] + parts[-1:]))
    for cand in candidates:
        try:
            return datetime.strptime(cand, "%a %b %d %H:%M:%S %Y")
        except Exception:
            pass
    return None

try:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            m = pat.match(line.rstrip("\n"))
            if not m:
                continue
            stamp = parse_stamp(m.group(2))
            if stamp is None:
                continue
            if m.group(1) == "Started at":
                started = stamp
            elif m.group(1) == "Terminated at":
                terminated = stamp
except Exception:
    print("0")
    raise SystemExit(0)

if started is None or terminated is None:
    print("0")
else:
    delta = int((terminated - started).total_seconds())
    print(str(delta if delta > 0 else 0))
PY
}

build_latest_log_status_map() {
  local dataset_dir="$1"
  local out_file="$2"
  local logs_dir="${dataset_dir}/logs"
  local latest_file classified_file
  local out_log base job type att

  : > "${out_file}"
  [[ -d "${logs_dir}" ]] || return 0

  latest_file="$(mktemp)"
  classified_file="$(mktemp)"

  for out_log in "${logs_dir}"/*.attempt*.out; do
    [[ -f "${out_log}" ]] || continue
    base="$(basename "${out_log}")"
    job=""
    type=""
    att=""

    if [[ "${base}" =~ ^(.+)\.export\.attempt([0-9]+)\.out$ ]]; then
      job="${BASH_REMATCH[1]}"
      att="${BASH_REMATCH[2]}"
      type="export"
    elif [[ "${base}" =~ ^(.+)\.attempt([0-9]+)\.out$ ]]; then
      job="${BASH_REMATCH[1]}"
      att="${BASH_REMATCH[2]}"
      type="segment"
    else
      continue
    fi

    printf "%s\t%s\t%s\t%s\n" "${job}" "${type}" "${att}" "${out_log}"
  done | sort -t $'\t' -k1,1 -k2,2 -k3,3n | awk -F'\t' '
    {
      key = $1 FS $2
      keep[key] = $0
    }
    END {
      for (k in keep) {
        print keep[k]
      }
    }
  ' > "${latest_file}"

  while IFS=$'\t' read -r job type att out_log; do
    [[ -n "${job}" ]] || continue
    local err_log state status stage_status elapsed_s
    err_log="${out_log%.out}.err"
    state="pending"
    status="pending"
    stage_status=""
    elapsed_s="$(parse_lsf_elapsed_seconds "${out_log}")"
    if ! [[ "${elapsed_s}" =~ ^[0-9]+$ ]]; then
      elapsed_s="0"
    fi

    if grep -q "Successfully completed" "${out_log}" 2>/dev/null; then
      state="done"
      status="ok"
      if [[ "${type}" == "segment" ]]; then
        stage_status="segment_ok"
      else
        stage_status="export_ok"
      fi
    elif grep -Eq "Exited with exit code|TERM_MEMLIMIT|TERM_RUNLIMIT|Request aborted by esub" "${out_log}" 2>/dev/null; then
      state="failed"
      status="lsf_exit"
      if grep -Eiq "TERM_MEMLIMIT|out of memory|cudaErrorMemoryAllocation|CUDA out of memory|std::bad_alloc" "${out_log}" "${err_log}" 2>/dev/null; then
        status="oom"
      elif grep -Eiq "CUDA_ERROR_ILLEGAL_ADDRESS|cudaErrorIllegalAddress|illegal memory access" "${out_log}" "${err_log}" 2>/dev/null; then
        status="gpu_illegal_access"
      elif grep -Eiq "No such file or directory" "${err_log}" 2>/dev/null; then
        status="missing_code_root"
      elif grep -Eiq "Request aborted by esub|does not contain a GPU reservation" "${out_log}" "${err_log}" 2>/dev/null; then
        status="submit_queue_mismatch"
      elif grep -Eiq "TERM_RUNLIMIT" "${out_log}" "${err_log}" 2>/dev/null; then
        status="runlimit"
      fi

      if [[ "${type}" == "segment" ]]; then
        case "${status}" in
          oom)
            stage_status="segment_oom"
            ;;
          runlimit)
            stage_status="segment_runlimit"
            ;;
          *)
            stage_status="segment_error"
            ;;
        esac
      else
        stage_status="export_error"
      fi
    elif grep -q "^Started at " "${out_log}" 2>/dev/null && ! grep -q "^Terminated at " "${out_log}" 2>/dev/null; then
      state="running"
      status="running"
      stage_status="running"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${job}" "${type}" "${att}" "${state}" "${status}" "${stage_status}" "${out_log}" "${elapsed_s}" >> "${classified_file}"
  done < "${latest_file}"

  awk -F'\t' -v OFS=$'\t' '
    {
      job = $1
      type = $2
      att = $3 + 0
      state = $4
      status = $5
      stage = $6
      log_file = $7
      elapsed_s = ($8 ~ /^[0-9]+$/) ? ($8 + 0) : 0

      jobs[job] = 1
      if (type == "segment") {
        seg_state[job] = state
        seg_status[job] = status
        seg_stage[job] = stage
        seg_runs[job] = att
        seg_log[job] = log_file
        seg_elapsed[job] = elapsed_s
      } else if (type == "export") {
        exp_state[job] = state
        exp_status[job] = status
        exp_stage[job] = stage
        exp_runs[job] = att
        exp_log[job] = log_file
        exp_elapsed[job] = elapsed_s
      }
    }
    END {
      for (job in jobs) {
        final_state = "pending"
        final_status = "pending"
        final_elapsed = 0
        run_count = seg_runs[job] + 0
        log_file = (seg_log[job] != "") ? seg_log[job] : exp_log[job]
        seg_stage_out = (seg_stage[job] != "") ? seg_stage[job] : "-"
        exp_stage_out = (exp_stage[job] != "") ? exp_stage[job] : "-"
        if (seg_elapsed[job] > 0) final_elapsed = seg_elapsed[job]
        else if (exp_elapsed[job] > 0) final_elapsed = exp_elapsed[job]

        if (seg_state[job] == "running") {
          final_state = "running"
          final_status = "running"
        } else if (seg_state[job] == "failed") {
          final_state = "failed"
          final_status = (seg_status[job] != "") ? seg_status[job] : "lsf_exit"
        } else if (seg_state[job] == "done") {
          final_state = "done"
          final_status = "ok"
          if (exp_state[job] == "running") {
            final_state = "running"
            final_status = "running"
          } else if (exp_state[job] == "failed") {
            final_state = "failed"
            final_status = (exp_status[job] != "") ? exp_status[job] : "export_error"
          }
        } else if (exp_state[job] == "running") {
          final_state = "running"
          final_status = "running"
        } else if (exp_state[job] == "failed") {
          final_state = "failed"
          final_status = (exp_status[job] != "") ? exp_status[job] : "export_error"
        }

        print job, final_state, final_status, run_count, log_file, seg_stage_out, exp_stage_out, final_elapsed
      }
    }
  ' "${classified_file}" > "${out_file}"

  rm -f "${latest_file}" "${classified_file}"
}

print_aggregate_header() {
  printf "dataset\tjob\tgroup\tgpu\tstatus\tstate\trunning\telapsed_s\trun_count\thad_rerun\thad_anc_retry\thad_predict_fallback\thad_recovery_pass\tseg_exists\tanndata_exists\txenium_exists\tseg_dir\tlog_file\tuse_3d\texpansion\ttx_max_k\ttx_max_dist\tn_mid_layers\tn_heads\tcells_min_counts\tmin_qv\talignment_loss\tsegment_status\texport_status\tsegment_job_id\texport_job_id\texport_note\tnote\trequested_queue\trequested_gmem_gb\trequested_mem_gb\trequested_wall\tmax_vram_mb\tmax_vram_gb\n"
}

append_normalized_snapshot_rows() {
  local dataset="$1"
  local dataset_dir="$2"
  local snapshot_file="$3"
  local log_status_map="$4"

  awk -F'\t' -v OFS=$'\t' -v dataset="${dataset}" -v plan_file="${dataset_dir}/job_plan.tsv" -v log_map_file="${log_status_map}" '
    function state_from_status(status, lower) {
      lower = tolower(status)
      if (status == "" || status == "-" || lower == "<none>") return "pending"
      if (lower ~ /running|in_progress/) return "running"
      if (lower == "partial") return "partial"
      if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
      if (lower ~ /^skipped_/ && lower != "skipped_existing") return "pending"
      if (lower == "pending" || lower == "planned") return "pending"
      if (lower == "reference") return "reference"
      return "failed"
    }
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
      return fallback
    }
    function canonical_bool(v, fallback, lower) {
      lower = tolower(v)
      if (lower == "1" || lower == "true" || lower == "yes") return "1"
      if (lower == "0" || lower == "false" || lower == "no") return "0"
      return fallback
    }
    function note_value(note, key, i, n, tok, parts) {
      n = split(note, parts, /[[:space:]]+/)
      for (i = 1; i <= n; i++) {
        tok = parts[i]
        if (index(tok, key "=") == 1) {
          return substr(tok, length(key) + 2)
        }
      }
      return ""
    }
    BEGIN {
      while ((getline log_line < log_map_file) > 0) {
        ln = split(log_line, l, "\t")
        if (ln < 8) continue
        ljob = l[1]
        if (ljob == "") continue
        log_state_map[ljob] = l[2]
        log_status_map[ljob] = l[3]
        log_run_count_map[ljob] = l[4]
        log_log_file_map[ljob] = l[5]
        log_segment_stage_map[ljob] = l[6]
        log_export_stage_map[ljob] = l[7]
        log_elapsed_s_map[ljob] = l[8]
      }
      close(log_map_file)

      plan_line = 0
      while ((getline line < plan_file) > 0) {
        n = split(line, p, "\t")
        if (plan_line == 0) {
          for (i = 1; i <= n; i++) {
            plan_idx[p[i]] = i
          }
          plan_line++
          continue
        }
        pj = (("job" in plan_idx) && plan_idx["job"] > 0 && plan_idx["job"] <= n) ? p[plan_idx["job"]] : ((n >= 1) ? p[1] : "")
        pg = (("group" in plan_idx) && plan_idx["group"] > 0 && plan_idx["group"] <= n) ? p[plan_idx["group"]] : ((n >= 2) ? p[2] : "")
        if (pj != "") group_by_job[pj] = pg
        plan_line++
      }
      close(plan_file)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        idx[$i] = i
      }
      next
    }
    NF == 0 { next }
    {
      job = col("job", $1)
      if (job == "" || job == "-") next

      group = col("group", "")
      if (group == "") group = group_by_job[job]
      if (group == "") group = "-"

      gpu = col("gpu", "-")

      status = col("status", "")
      if (status == "" || status == "-") status = "pending"

      state = col("state", "")
      if (state == "" || state == "-") state = state_from_status(status)

      running = col("running", "")
      if (running == "" || running == "-") {
        running = (tolower(state) == "running") ? "1" : "0"
      }
      running = canonical_bool(running, "0")

      elapsed_s = col("elapsed_s", col("elapsed", ""))
      if (elapsed_s == "" || elapsed_s == "-") elapsed_s = "0"

      run_count = col("run_count", "")
      if (run_count == "" || run_count == "-") run_count = "0"

      had_rerun = canonical_bool(col("had_rerun", ""), "0")
      had_anc_retry = canonical_bool(col("had_anc_retry", ""), "0")
      had_predict_fallback = canonical_bool(col("had_predict_fallback", ""), "0")
      had_recovery_pass = canonical_bool(col("had_recovery_pass", ""), "0")

      seg_exists = canonical_bool(col("seg_exists", ""), "0")
      anndata_exists = canonical_bool(col("anndata_exists", ""), "0")
      xenium_exists = canonical_bool(col("xenium_exists", ""), "0")

      seg_dir = col("seg_dir", "")
      if (seg_dir == "" && NF >= 6) seg_dir = $6
      if (seg_dir == "") seg_dir = "-"

      log_file = col("log_file", "")
      if (log_file == "" && NF >= 7) log_file = $7
      if (log_file == "") log_file = "-"

      use_3d = col("use_3d", "-")
      expansion = col("expansion", "-")
      tx_max_k = col("tx_max_k", "-")
      tx_max_dist = col("tx_max_dist", "-")
      n_mid_layers = col("n_mid_layers", "-")
      n_heads = col("n_heads", "-")
      cells_min_counts = col("cells_min_counts", "-")
      min_qv = col("min_qv", "-")
      alignment_loss = col("alignment_loss", "-")
      segment_status = col("segment_status", "-")
      export_status = col("export_status", "-")
      segment_job_id = col("segment_job_id", "-")
      export_job_id = col("export_job_id", "-")
      export_note = col("export_note", "-")
      note = col("note", "-")
      requested_queue = col("requested_queue", "")
      requested_gmem_gb = col("requested_gmem_gb", "")
      requested_mem_gb = col("requested_mem_gb", "")
      requested_wall = col("requested_wall", "")
      max_vram_mb = col("max_vram_mb", "")
      max_vram_gb = col("max_vram_gb", "")

      if (segment_status == "") segment_status = "-"
      if (export_status == "") export_status = "-"
      if (segment_job_id == "") segment_job_id = "-"
      if (export_job_id == "") export_job_id = "-"
      if (export_note == "") export_note = "-"
      if (note == "") note = "-"
      if (requested_queue == "") requested_queue = note_value(note, "queue")
      if (requested_gmem_gb == "") {
        requested_gmem_gb = note_value(note, "gmem")
        sub(/G$/, "", requested_gmem_gb)
      }
      if (requested_mem_gb == "") {
        requested_mem_gb = note_value(note, "mem")
        sub(/G$/, "", requested_mem_gb)
      }
      if (requested_wall == "") requested_wall = note_value(note, "wall")
      if (requested_queue == "") requested_queue = "-"
      if (requested_gmem_gb == "") requested_gmem_gb = "-"
      if (requested_mem_gb == "") requested_mem_gb = "-"
      if (requested_wall == "") requested_wall = "-"
      if (max_vram_mb == "") max_vram_mb = "-"
      if (max_vram_gb == "") max_vram_gb = "-"

      log_state = (job in log_state_map) ? log_state_map[job] : ""
      log_status = (job in log_status_map) ? log_status_map[job] : ""
      log_runs = (job in log_run_count_map) ? log_run_count_map[job] : ""
      log_log_file = (job in log_log_file_map) ? log_log_file_map[job] : ""
      log_seg_stage = (job in log_segment_stage_map) ? log_segment_stage_map[job] : ""
      log_exp_stage = (job in log_export_stage_map) ? log_export_stage_map[job] : ""
      log_elapsed_s = (job in log_elapsed_s_map) ? log_elapsed_s_map[job] : ""

      if (log_state != "") {
        lower_status = tolower(status)
        if (status == "" || status == "-" || lower_status == "pending" || lower_status == "planned" || lower_status == "running" || lower_status == "in_progress") {
          state = log_state
          if (log_status != "") status = log_status
        }
        if (log_runs ~ /^[0-9]+$/) {
          if (run_count !~ /^[0-9]+$/) run_count = "0"
          if ((log_runs + 0) > (run_count + 0)) run_count = log_runs
        }
        if ((log_file == "" || log_file == "-") && log_log_file != "") log_file = log_log_file
        if (log_elapsed_s ~ /^[0-9]+$/ && (log_elapsed_s + 0) > 0) {
          elapsed_s = log_elapsed_s
        }
        if (tolower(log_state) == "failed" && log_log_file != "") {
          fail_log = log_log_file
          sub(/\.out$/, ".err", fail_log)
          log_file = fail_log
        }

        lower_segment_status = tolower(segment_status)
        if ((segment_status == "" || segment_status == "-" || lower_segment_status == "pending" || lower_segment_status == "planned" || lower_segment_status == "running") && log_seg_stage != "" && log_seg_stage != "-") {
          segment_status = log_seg_stage
        }
        lower_export_status = tolower(export_status)
        if ((export_status == "" || export_status == "-" || lower_export_status == "pending" || lower_export_status == "planned" || lower_export_status == "running") && log_exp_stage != "" && log_exp_stage != "-") {
          export_status = log_exp_stage
        }

        if (log_exp_stage == "export_error") {
          state = "failed"
          if (log_status != "") status = log_status
          export_status = "export_error"
        }
      }

      lower_status = tolower(status)
      if (lower_status ~ /^skipped_/ && lower_status != "skipped_existing") {
        state = "pending"
        running = "0"
      }
      if (state == "" || state == "-") state = state_from_status(status)
      if (run_count ~ /^[0-9]+$/ && (run_count + 0) > 1) had_rerun = "1"

      # Segmentation availability is the source of truth for completion.
      if (seg_exists == "1") {
        state = "done"
        running = "0"
        status = (anndata_exists == "1") ? "ok" : "segment_only"
      }

      print dataset, job, group, gpu, status, state, running, elapsed_s, run_count, had_rerun, had_anc_retry,
            had_predict_fallback, had_recovery_pass, seg_exists, anndata_exists, xenium_exists, seg_dir,
            log_file, use_3d, expansion, tx_max_k, tx_max_dist, n_mid_layers, n_heads, cells_min_counts,
            min_qv, alignment_loss, segment_status, export_status, segment_job_id, export_job_id, export_note, note,
            requested_queue, requested_gmem_gb, requested_mem_gb, requested_wall, max_vram_mb, max_vram_gb
    }
  ' "${snapshot_file}" >> "${OUT_TSV}"
}

build_aggregate_snapshot() {
  local dataset_dir dataset snapshot all_jobs log_status_map found_any=0
  local all_jobs_fallback_count=0

  DASH_WARNINGS=()
  print_aggregate_header > "${OUT_TSV}"

  for dataset_dir in "${DATASET_DIRS[@]}"; do
    [[ -d "${dataset_dir}" ]] || continue
    found_any=1
    dataset="$(basename "${dataset_dir}")"

    if [[ "${REFRESH}" == "1" ]]; then
      if ! refresh_dataset_snapshot "${dataset_dir}"; then
        add_warning "refresh failed for dataset ${dataset}"
      fi
    fi

    snapshot="${dataset_dir}/summaries/status_snapshot.tsv"
    all_jobs="${dataset_dir}/summaries/all_jobs.tsv"
    log_status_map="$(mktemp)"
    build_latest_log_status_map "${dataset_dir}" "${log_status_map}"

    if [[ -f "${snapshot}" ]]; then
      append_normalized_snapshot_rows "${dataset}" "${dataset_dir}" "${snapshot}" "${log_status_map}"
    elif [[ -f "${all_jobs}" ]]; then
      all_jobs_fallback_count=$((all_jobs_fallback_count + 1))
      append_normalized_snapshot_rows "${dataset}" "${dataset_dir}" "${all_jobs}" "${log_status_map}"
    else
      add_warning "no snapshot/all_jobs found for dataset ${dataset}"
    fi
    rm -f "${log_status_map}"
  done

  if [[ "${all_jobs_fallback_count}" -gt 0 ]]; then
    add_warning "using all_jobs.tsv fallback for ${all_jobs_fallback_count} dataset(s)"
  fi

  if [[ "${found_any}" == "0" ]]; then
    add_warning "No dataset roots found under ${ROOT}/datasets or ${ROOT}"
  fi
}

collect_live_lsf_state_map() {
  local out_file="$1"
  local id_map_file state_rows_file raw_file id_list submit_host raw_jobs_file
  local dataset job selected_id log_file fallback_id

  : > "${out_file}"
  id_map_file="$(mktemp)"
  state_rows_file="$(mktemp)"
  raw_file="$(mktemp)"
  raw_jobs_file="$(mktemp)"

  awk -F'\t' -v OFS=$'\t' '
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0 && idx[name] <= NF) return $(idx[name])
      return fallback
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = col("dataset", "")
      job = col("job", "")
      seg_id = col("segment_job_id", "")
      exp_id = col("export_job_id", "")
      exp_status = tolower(col("export_status", ""))
      note = col("note", "")
      log_file = col("log_file", "")
      selected = ""

      if (dataset == "" || job == "") next
      if (exp_id != "" && exp_id != "-" && (exp_status == "planned" || exp_status == "pending" || exp_status == "running")) {
        selected = exp_id
      } else if (seg_id != "" && seg_id != "-") {
        selected = seg_id
      }
      if ((selected == "" || selected == "-") && match(note, /lsf_job=[0-9]+/)) {
        selected = substr(note, RSTART + 8, RLENGTH - 8)
      }

      gsub(/[^0-9]/, "", selected)
      print dataset, job, selected, log_file
    }
  ' "${OUT_TSV}" > "${raw_jobs_file}"

  : > "${id_map_file}"
  while IFS=$'\t' read -r dataset job selected_id log_file; do
    [[ -n "${dataset}" && -n "${job}" ]] || continue
    fallback_id="${selected_id}"
    if [[ -z "${fallback_id}" || "${fallback_id}" == "-" ]]; then
      if [[ -n "${log_file}" && "${log_file}" != "-" && -f "${log_file}" ]]; then
        fallback_id="$(awk -F'lsf_job=' '
          /START job=/ {
            if (NF >= 2) {
              split($2, a, /[^0-9]/)
              if (a[1] != "") id = a[1]
            }
          }
          END {
            if (id != "") print id
          }
        ' "${log_file}" 2>/dev/null || true)"
      fi
    fi
    fallback_id="$(printf '%s' "${fallback_id}" | tr -cd '0-9')"
    if [[ -n "${fallback_id}" ]]; then
      printf "%s\t%s\t%s\n" "${dataset}" "${job}" "${fallback_id}" >> "${id_map_file}"
    fi
  done < "${raw_jobs_file}"

  if [[ ! -s "${id_map_file}" ]]; then
    rm -f "${id_map_file}" "${state_rows_file}" "${raw_file}" "${raw_jobs_file}"
    return 0
  fi

  id_list="$(awk -F'\t' '!seen[$3]++ { printf("%s%s", sep, $3); sep = " " }' "${id_map_file}")"
  if [[ -z "${id_list}" ]]; then
    rm -f "${id_map_file}" "${state_rows_file}" "${raw_file}" "${raw_jobs_file}"
    return 0
  fi

  submit_host="${LSF_SUBMIT_HOST:-}"
  case "${submit_host}" in
    ""|local|LOCAL|none|NONE|-)
      submit_host=""
      ;;
  esac

  if [[ -n "${submit_host}" ]]; then
    if command -v ssh >/dev/null 2>&1; then
      ssh -o BatchMode=yes "${submit_host}" "bjobs -a -noheader -o 'jobid stat' ${id_list}" > "${raw_file}" 2>/dev/null || true
    else
      add_warning "ssh not available; live RUN/PEND overlay disabled (LSF_SUBMIT_HOST=${submit_host})"
    fi
  elif command -v bjobs >/dev/null 2>&1; then
    bjobs -a -noheader -o "jobid stat" ${id_list} > "${raw_file}" 2>/dev/null || true
  else
    add_warning "bjobs unavailable; run on submit host or set LSF_SUBMIT_HOST for live RUN/PEND states"
  fi

  awk '
    NF >= 2 {
      print $1 "\t" $2
    }
  ' "${raw_file}" > "${state_rows_file}"

  if [[ ! -s "${state_rows_file}" ]]; then
    rm -f "${id_map_file}" "${state_rows_file}" "${raw_file}" "${raw_jobs_file}"
    return 0
  fi

  awk -F'\t' -v OFS=$'\t' '
    NR == FNR {
      state_by_id[$1] = $2
      next
    }
    {
      state = ($3 in state_by_id) ? state_by_id[$3] : ""
      print $1, $2, $3, state
    }
  ' "${state_rows_file}" "${id_map_file}" > "${out_file}"

  rm -f "${id_map_file}" "${state_rows_file}" "${raw_file}" "${raw_jobs_file}"
}

apply_live_lsf_state_overlay() {
  local live_map_file tmp_file
  live_map_file="$(mktemp)"
  collect_live_lsf_state_map "${live_map_file}"
  if [[ ! -s "${live_map_file}" ]]; then
    rm -f "${live_map_file}"
    return 0
  fi

  tmp_file="$(mktemp)"
  awk -F'\t' -v OFS=$'\t' -v map_file="${live_map_file}" '
    BEGIN {
      while ((getline line < map_file) > 0) {
        n = split(line, m, "\t")
        if (n < 4) continue
        key = m[1] SUBSEP m[2]
        live_state[key] = m[4]
      }
      close(map_file)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      print
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      key = dataset SUBSEP job
      lsf = (key in live_state) ? live_state[key] : ""

      state = (("state" in idx) && idx["state"] > 0) ? $(idx["state"]) : "pending"
      status = (("status" in idx) && idx["status"] > 0) ? $(idx["status"]) : "pending"
      running = (("running" in idx) && idx["running"] > 0) ? $(idx["running"]) : "0"
      lower_state = tolower(state)
      lower_status = tolower(status)

      if (lsf == "RUN" || lsf == "PROV") {
        state = "running"
        status = "running"
        running = "1"
      } else if (lsf == "PEND" || lsf == "PSUSP" || lsf == "USUSP" || lsf == "SSUSP" || lsf == "WAIT") {
        if (lower_state != "done") {
          state = "pending"
          if (lower_status == "" || lower_status == "-" || lower_status == "running" || lower_status == "in_progress" || lower_status == "pending" || lower_status == "planned") {
            status = "pending"
          }
        }
        running = "0"
      } else if (lsf == "EXIT" || lsf == "ZOMBI" || lsf == "UNKWN") {
        if (lower_state != "done") {
          state = "failed"
          if (lower_status == "" || lower_status == "-" || lower_status == "running" || lower_status == "in_progress" || lower_status == "pending" || lower_status == "planned") {
            status = "lsf_exit"
          }
        }
        running = "0"
      } else if (lsf == "DONE") {
        running = "0"
      }

      if (("state" in idx) && idx["state"] > 0) $(idx["state"]) = state
      if (("status" in idx) && idx["status"] > 0) $(idx["status"]) = status
      if (("running" in idx) && idx["running"] > 0) $(idx["running"]) = running
      print
    }
  ' "${OUT_TSV}" > "${tmp_file}"

  mv "${tmp_file}" "${OUT_TSV}"
  rm -f "${live_map_file}"
}

build_dataset_summary() {
  local out_file="$1"
  local sorted_file
  sorted_file="$(mktemp)"

  awk -F'\t' -v OFS=$'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (dataset == "" || job == "") next

      jobs[dataset]++
      if (state == "running") running[dataset]++
      else if (state == "pending") pending[dataset]++
      else if (state == "failed") failed[dataset]++
      else if (state == "done") done[dataset]++
      else if (state == "partial") partial[dataset]++
      else other[dataset]++
    }
    END {
      print "dataset\tjobs\trunning\tpending\tfailed\tdone\tpartial\tother\tprocessed\tprogress_pct"
      for (dataset in jobs) {
        processed = done[dataset] + partial[dataset] + failed[dataset]
        pct = (jobs[dataset] > 0) ? (processed * 100.0 / jobs[dataset]) : 0
        printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\n",
          dataset,
          jobs[dataset] + 0,
          running[dataset] + 0,
          pending[dataset] + 0,
          failed[dataset] + 0,
          done[dataset] + 0,
          partial[dataset] + 0,
          other[dataset] + 0,
          processed,
          pct
      }
    }
  ' "${OUT_TSV}" > "${sorted_file}"

  {
    header="$(head -n 1 "${sorted_file}" 2>/dev/null || true)"
    if [[ -n "${header}" ]]; then
      printf "%s\n" "${header}"
    fi
    tail -n +2 "${sorted_file}" | sort -t $'\t' -k5,5nr -k3,3nr -k4,4nr -k1,1
  } > "${out_file}"

  rm -f "${sorted_file}"
}

build_dataset_tx_rows_map() {
  local out_file="$1"
  local seg_map_file
  local dataset seg_dir tx_file tx_rows

  : > "${out_file}"
  seg_map_file="$(mktemp)"

  awk -F'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      seg_dir = (("seg_dir" in idx) && idx["seg_dir"] > 0) ? $(idx["seg_dir"]) : ""
      if (dataset == "" || seg_dir == "" || seg_dir == "-") next

      if (!(dataset in any_seg_dir)) any_seg_dir[dataset] = seg_dir
      if (job == "baseline" && !(dataset in baseline_seg_dir)) baseline_seg_dir[dataset] = seg_dir
    }
    END {
      for (dataset in any_seg_dir) {
        seg_dir = (dataset in baseline_seg_dir) ? baseline_seg_dir[dataset] : any_seg_dir[dataset]
        print dataset "\t" seg_dir
      }
    }
  ' "${OUT_TSV}" > "${seg_map_file}"

  while IFS=$'\t' read -r dataset seg_dir; do
    [[ -n "${dataset}" ]] || continue
    tx_rows="-"
    tx_file="${seg_dir}/transcripts.parquet"
    if [[ -r "${tx_file}" ]]; then
      tx_rows="$(parquet_row_count_fast "${tx_file}")"
      [[ -n "${tx_rows}" ]] || tx_rows="-"
    fi
    printf "%s\t%s\n" "${dataset}" "${tx_rows}" >> "${out_file}"
  done < "${seg_map_file}"

  rm -f "${seg_map_file}"
}

build_dataset_overview_minimal() {
  local out_file="$1"
  local tx_map_file="$2"
  local sorted_file
  sorted_file="$(mktemp)"

  awk -F'\t' -v OFS=$'\t' -v tx_map_file="${tx_map_file}" \
      -v c_green="${C_GREEN}" -v c_yellow="${C_YELLOW}" -v c_red="${C_RED}" -v c_reset="${C_RESET}" '
    function repeat_char(ch, n, out, i) {
      out = ""
      for (i = 0; i < n; i++) out = out ch
      return out
    }
    function progress_bar(done_cnt, total, width, filled, empty, left, right) {
      width = 18
      if (total <= 0) return "[" repeat_char("-", width) "] 0/0"
      filled = int((done_cnt * width) / total)
      if (filled < 0) filled = 0
      if (filled > width) filled = width
      empty = width - filled
      left = repeat_char("#", filled)
      right = repeat_char("-", empty)
      return "[" left right "] " done_cnt "/" total
    }
    BEGIN {
      while ((getline line < tx_map_file) > 0) {
        n = split(line, t, "\t")
        if (n < 1 || t[1] == "") continue
        tx_rows[t[1]] = (n >= 2 && t[2] != "") ? t[2] : "-"
      }
      close(tx_map_file)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      elapsed_s = (("elapsed_s" in idx) && idx["elapsed_s"] > 0) ? $(idx["elapsed_s"]) : "0"
      run_count = (("run_count" in idx) && idx["run_count"] > 0) ? $(idx["run_count"]) : "0"
      if (dataset == "") next

      jobs[dataset]++
      if (run_count ~ /^[0-9]+$/ && (run_count + 0) > 1) rerun_attempts[dataset] += ((run_count + 0) - 1)
      if (state == "done") {
        done[dataset]++
        if (elapsed_s ~ /^[0-9]+(\.[0-9]+)?$/) {
          done_elapsed_sum[dataset] += (elapsed_s + 0.0)
          done_elapsed_n[dataset]++
        }
      } else if (state == "failed") {
        failed[dataset]++
      } else if (state == "running") {
        running[dataset]++
      } else if (state == "pending") {
        pending[dataset]++
      } else if (state == "partial") {
        partial[dataset]++
      } else {
        pending[dataset]++
      }
    }
    END {
      print "rank\tdataset\tprogress\tdone\trunning\tpending\tfailed\tattempts_2plus\tavg_gpu_h_done\ttx_rows"
      for (dataset in jobs) {
        running_cnt = running[dataset] + 0
        pending_cnt = pending[dataset] + 0
        done_cnt = done[dataset] + 0
        failed_cnt = failed[dataset] + 0
        total = jobs[dataset] + 0
        avg_gpu_h_done = "-"
        progress_col = progress_bar(done_cnt, total)
        progress_color = c_yellow
        if (total > 0 && done_cnt == total) progress_color = c_green
        else if (total > 0 && failed_cnt == total) progress_color = c_red
        else if (done_cnt == 0 && pending_cnt == 0 && running_cnt == 0 && failed_cnt > 0) progress_color = c_red
        if (progress_color != "") progress_col = progress_color progress_col c_reset
        if ((done_elapsed_n[dataset] + 0) > 0) {
          avg_gpu_h_done = sprintf("%.2f", (done_elapsed_sum[dataset] / 3600.0) / done_elapsed_n[dataset])
        }
        tx = (dataset in tx_rows) ? tx_rows[dataset] : "-"
        print failed_cnt, dataset, progress_col, done_cnt, running_cnt, pending_cnt, failed_cnt, (rerun_attempts[dataset] + 0), avg_gpu_h_done, tx
      }
    }
  ' "${OUT_TSV}" > "${sorted_file}"

  {
    header="$(head -n 1 "${sorted_file}" 2>/dev/null || true)"
    if [[ -n "${header}" ]]; then
      printf "dataset\tprogress\tdone\trunning\tpending\tfailed\tattempts_2plus\tavg_gpu_h_done\ttx_rows\n"
    fi
    tail -n +2 "${sorted_file}" | sort -t $'\t' -k1,1nr -k5,5nr -k6,6nr -k2,2 | cut -f2-
  } > "${out_file}"

  rm -f "${sorted_file}"
}

build_state_counts() {
  local out_file="$1"
  awk -F'\t' -v OFS=$'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (state == "") state = "pending"
      c[state]++
    }
    END {
      print "state\tcount"
      order[1] = "running"
      order[2] = "pending"
      order[3] = "failed"
      order[4] = "done"
      order[5] = "partial"
      for (i = 1; i <= 5; i++) {
        s = order[i]
        print s, c[s] + 0
        seen[s] = 1
      }
      for (k in c) {
        if (!(k in seen)) print k, c[k]
      }
    }
  ' "${OUT_TSV}" > "${out_file}"
}

build_status_counts() {
  local out_file="$1"
  awk -F'\t' -v OFS=$'\t' '
    function state_from_status(status, lower) {
      lower = tolower(status)
      if (status == "" || status == "-" || lower == "<none>") return "pending"
      if (lower ~ /running|in_progress/) return "running"
      if (lower == "partial") return "partial"
      if (lower == "ok" || lower == "skipped_existing" || lower == "recovered_predict_ok") return "done"
      if (lower ~ /^skipped_/ && lower != "skipped_existing") return "pending"
      if (lower == "pending" || lower == "planned") return "pending"
      if (lower == "reference") return "reference"
      return "failed"
    }
    function rank(state, lower) {
      lower = tolower(state)
      if (lower == "running") return 1
      if (lower == "pending") return 2
      if (lower == "failed") return 3
      if (lower == "done") return 4
      if (lower == "partial") return 5
      if (lower == "reference") return 6
      return 99
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      status = (("status" in idx) && idx["status"] > 0) ? $(idx["status"]) : ""
      if (status == "") status = "<none>"
      c[status]++
    }
    END {
      print "rank\tstatus\tcount"
      for (k in c) {
        st = state_from_status(k)
        print rank(st), k, c[k]
      }
    }
  ' "${OUT_TSV}" | {
    IFS= read -r header || true
    if [[ -n "${header}" ]]; then
      printf "status\tcount\n"
    fi
    sort -t $'\t' -k1,1n -k3,3nr -k2,2 | cut -f2-
  } > "${out_file}"
}

build_job_overview() {
  local out_file="$1"
  local sorted_file
  local root_abs=""
  sorted_file="$(mktemp)"
  if [[ -d "${ROOT}" ]]; then
    root_abs="$(cd "${ROOT}" && pwd)/"
  fi

  awk -F'\t' -v OFS=$'\t' -v root_prefix="${root_abs}" -v root_rel_prefix="${ROOT%/}/" '
    function col(name, fallback) {
      if ((name in idx) && idx[name] > 0) return $(idx[name])
      return fallback
    }
    function shorten_path(p) {
      if (p == "" || p == "-") return "-"
      if (root_prefix != "" && index(p, root_prefix) == 1) {
        return substr(p, length(root_prefix) + 1)
      }
      if (root_rel_prefix != "" && index(p, root_rel_prefix) == 1) {
        return substr(p, length(root_rel_prefix) + 1)
      }
      return p
    }
    function state_rank(state, lower) {
      lower = tolower(state)
      if (lower == "failed") return 1
      if (lower == "running") return 2
      if (lower == "partial") return 3
      if (lower == "pending") return 4
      if (lower == "done") return 5
      if (lower == "reference") return 6
      return 99
    }
    function fmt_minutes(v, lower, n) {
      lower = tolower(v)
      if (v == "" || v == "-" || lower == "nan" || lower == "none") return "0.00"
      n = (v + 0.0) / 60.0
      if (n < 0) n = 0
      return sprintf("%.2f", n)
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      print "rank\tdataset\tjob\tgroup\tgpu\tstate\tstatus\treq_queue\treq_gmem_gb\treq_mem_gb\treq_wall\tmax_vram_gb\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\tlog_file"
      next
    }
    {
      dataset = col("dataset", "-")
      job = col("job", "-")
      state = tolower(col("state", "pending"))
      status = col("status", "pending")
      lower_status = tolower(status)
      display_status = status
      if (dataset == "" || job == "") next
      if (lower_status == "oom" || lower_status == "segment_oom") {
        display_status = "oom_memlimit"
      }

      print state_rank(state),
            dataset,
            job,
            col("group", "-"),
            col("gpu", "-"),
            state,
            display_status,
            col("requested_queue", "-"),
            col("requested_gmem_gb", "-"),
            col("requested_mem_gb", "-"),
            col("requested_wall", "-"),
            col("max_vram_gb", "-"),
            col("run_count", "0"),
            fmt_minutes(col("elapsed_s", "0")),
            col("had_rerun", "0"),
            col("had_anc_retry", "0"),
            col("had_predict_fallback", "0"),
            col("had_recovery_pass", "0"),
            col("seg_exists", "0"),
            col("anndata_exists", "0"),
            col("xenium_exists", "0"),
            shorten_path(col("log_file", "-"))
    }
  ' "${OUT_TSV}" > "${sorted_file}"

  {
    header="$(head -n 1 "${sorted_file}" 2>/dev/null || true)"
    if [[ -n "${header}" ]]; then
      printf "dataset\tjob\tgroup\tgpu\tstate\tstatus\treq_queue\treq_gmem_gb\treq_mem_gb\treq_wall\tmax_vram_gb\truns\telapsed_min\trerun\tanc_retry\toom_pred_fallback\trecovery\tseg\tanndata\txenium\tlog_file\n"
    fi
    tail -n +2 "${sorted_file}" | sort -t $'\t' -k1,1n -k2,2 -k3,3 | cut -f2-
  } > "${out_file}"

  rm -f "${sorted_file}"
}

build_recent_runs() {
  local out_file="$1"
  local raw_file
  local dataset_dir dataset out_log job log_name
  raw_file="$(mktemp)"

  printf "sort_key\tdataset\tjob\tlsf_job\texit\tsegger_rc\tanndata\tcompleted\tstarted_at\tterminated_at\tout_log\n" > "${raw_file}"

  for dataset_dir in "${DATASET_DIRS[@]}"; do
    [[ -d "${dataset_dir}" ]] || continue
    dataset="$(basename "${dataset_dir}")"
    for out_log in "${dataset_dir}"/logs/*.attempt*.out; do
      [[ -f "${out_log}" ]] || continue
      log_name="$(basename "${out_log}")"
      job="${log_name%%.attempt*}"

      awk -v dataset="${dataset}" -v job="${job}" -v out_log="${out_log}" '
        function month_num(mon) {
          if (mon == "Jan") return 1
          if (mon == "Feb") return 2
          if (mon == "Mar") return 3
          if (mon == "Apr") return 4
          if (mon == "May") return 5
          if (mon == "Jun") return 6
          if (mon == "Jul") return 7
          if (mon == "Aug") return 8
          if (mon == "Sep") return 9
          if (mon == "Oct") return 10
          if (mon == "Nov") return 11
          if (mon == "Dec") return 12
          return 0
        }
        function make_key(mon, day, time, year, mm) {
          mm = month_num(mon)
          if (mm == 0 || time == "" || year == "") return ""
          return sprintf("%04d-%02d-%02d %s", year + 0, mm, day + 0, time)
        }
        function flush(sort_key) {
          if (!in_block || job_id == "") return
          sort_key = start_key
          if (sort_key == "") sort_key = end_key
          if (sort_key == "") sort_key = "0000-00-00 00:00:00"
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
            sort_key,
            dataset,
            job,
            job_id,
            exit_code,
            segger_rc,
            wrote_anndata ? "1" : "0",
            completed,
            started_at,
            terminated_at,
            out_log
        }
        /^Subject: Job / {
          flush()
          in_block = 1
          job_id = $3
          sub(/:$/, "", job_id)
          exit_code = ""
          segger_rc = ""
          wrote_anndata = 0
          completed = 0
          started_at = ""
          terminated_at = ""
          start_key = ""
          end_key = ""
          next
        }
        in_block && /^Started at / {
          started_at = $0
          sub(/^Started at /, "", started_at)
          start_key = make_key($4, $5, $6, $7)
          next
        }
        in_block && /^Terminated at / {
          terminated_at = $0
          sub(/^Terminated at /, "", terminated_at)
          end_key = make_key($4, $5, $6, $7)
          next
        }
        in_block && /^Exited with exit code / {
          exit_code = $5
          sub(/\.$/, "", exit_code)
          next
        }
        in_block && /^\[JOB\] segger_rc=/ {
          segger_rc = $0
          sub(/^\[JOB\] segger_rc=/, "", segger_rc)
          next
        }
        in_block && /^\[JOB\] completed$/ {
          completed = 1
          next
        }
        in_block && /^Written AnnData output:/ {
          wrote_anndata = 1
          next
        }
        END {
          flush()
        }
      ' "${out_log}" >> "${raw_file}"
    done
  done

  {
    printf "dataset\tjob\tlsf_job\texit\tsegger_rc\tanndata\tcompleted\tstarted_at\tterminated_at\tout_log\n"
    if [[ "${RECENT_LIMIT}" == "0" ]]; then
      tail -n +2 "${raw_file}" | sort -t $'\t' -k1,1 -k2,2 -k3,3 | cut -f2-
    else
      tail -n +2 "${raw_file}" | sort -t $'\t' -k1,1 -k2,2 -k3,3 | tail -n "${RECENT_LIMIT}" | cut -f2-
    fi
  } > "${out_file}"

  rm -f "${raw_file}"
}

render_once() {
  local snapshot="${OUT_TSV}"
  local counts dataset_count total_jobs done_count running_count pending_count partial_count failed_count
  local processed
  local dataset_summary_file state_counts_file status_counts_file job_overview_file recent_runs_file
  local dataset_overview_min_file dataset_tx_rows_file
  local done_jobs_file done_jobs_compact_file failed_jobs_file failed_jobs_compact_file

  build_aggregate_snapshot
  apply_live_lsf_state_overlay

  counts="$(awk -F'\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) idx[$i] = i
      next
    }
    {
      dataset = (("dataset" in idx) && idx["dataset"] > 0) ? $(idx["dataset"]) : ""
      job = (("job" in idx) && idx["job"] > 0) ? $(idx["job"]) : ""
      state = (("state" in idx) && idx["state"] > 0) ? tolower($(idx["state"])) : "pending"
      if (dataset == "" || job == "") next

      seen[dataset] = 1
      jobs++
      if (state == "done") done++
      else if (state == "running") running++
      else if (state == "pending") pending++
      else if (state == "partial") partial++
      else if (state == "failed") failed++
      else pending++
    }
    END {
      for (dataset in seen) datasets++
      printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
        datasets + 0,
        jobs + 0,
        done + 0,
        running + 0,
        pending + 0,
        partial + 0,
        failed + 0
    }
  ' "${snapshot}")"

  IFS=$'\t' read -r dataset_count total_jobs done_count running_count pending_count partial_count failed_count <<< "${counts}"
  processed=$((done_count + partial_count + failed_count))

  dataset_summary_file="$(mktemp)"
  state_counts_file="$(mktemp)"
  status_counts_file="$(mktemp)"
  job_overview_file="$(mktemp)"
  recent_runs_file="$(mktemp)"
  dataset_overview_min_file="$(mktemp)"
  dataset_tx_rows_file="$(mktemp)"
  done_jobs_file=""
  done_jobs_compact_file=""
  failed_jobs_file=""
  failed_jobs_compact_file=""

  build_dataset_summary "${dataset_summary_file}"
  build_state_counts "${state_counts_file}"
  build_status_counts "${status_counts_file}"
  build_job_overview "${job_overview_file}"
  build_recent_runs "${recent_runs_file}"
  build_dataset_tx_rows_map "${dataset_tx_rows_file}"
  build_dataset_overview_minimal "${dataset_overview_min_file}" "${dataset_tx_rows_file}"

  if [[ "${MINIMAL_MODE}" == "1" ]]; then
    printf "%b\n" "${C_CYAN}Aggregate Benchmark Dashboard (Minimal)${C_RESET} | $(timestamp)"
    printf "Root: %s\n" "${ROOT}"
    printf "Snapshot: %s\n" "${snapshot}"
    printf "Datasets: %d  Jobs: %d\n" "${dataset_count}" "${total_jobs}"
    printf "Progress: %s\n" "$(draw_progress_bar "${processed}" "${total_jobs}")"
    printf "%b\n" "${C_BLUE}running=${running_count}${C_RESET}  ${C_YELLOW}pending=${pending_count}${C_RESET}  ${C_RED}failed=${failed_count}${C_RESET}  ${C_GREEN}done=${done_count}${C_RESET}  ${C_CYAN}partial=${partial_count}${C_RESET}"

    if [[ "${#DASH_WARNINGS[@]}" -gt 0 ]]; then
      echo
      printf "%b\n" "${C_YELLOW}Warnings:${C_RESET}"
      for warning in "${DASH_WARNINGS[@]}"; do
        printf -- "- %s\n" "${warning}"
      done
    fi

    echo
    echo "Dataset Overview:"
    render_tsv_table < "${dataset_overview_min_file}"

    echo
    done_jobs_file="$(mktemp)"
    done_jobs_compact_file="$(mktemp)"
    failed_jobs_file="$(mktemp)"
    failed_jobs_compact_file="$(mktemp)"
    awk -F'\t' 'NR == 1 || tolower($5) == "done" { print }' "${job_overview_file}" > "${done_jobs_file}"
    awk -F'\t' '
      function bool_label(v, lower) {
        lower = tolower(v)
        if (v == "1" || lower == "true" || lower == "yes") return "yes"
        return "no"
      }
      NR == 1 {
        print "dataset\tattempts\tjob\tgpu_time_h\telapsed_min\tmax_vram_gb\tseg\tanndata"
        next
      }
      {
        elapsed_min = $13
        elapsed_num = elapsed_min + 0.0
        gpu_time_h = sprintf("%.2f", elapsed_num / 60.0)
        print $1 "\t" $12 "\t" $2 "\t" gpu_time_h "\t" elapsed_min "\t" $11 "\t" bool_label($18) "\t" bool_label($19)
      }
    ' "${done_jobs_file}" > "${done_jobs_compact_file}"
    awk -F'\t' 'NR == 1 || tolower($5) == "failed" { print }' "${job_overview_file}" > "${failed_jobs_file}"
    awk -F'\t' '
      NR == 1 {
        print "dataset\tattempts\tjob\telapsed_min\treason\tlog_path"
        next
      }
      {
        print $1 "\t" $12 "\t" $2 "\t" $13 "\t" $6 "\t" $21
      }
    ' "${failed_jobs_file}" > "${failed_jobs_compact_file}"

    echo "Done Jobs:"
    colorize_rows_single_color "${C_GREEN}" < "${done_jobs_compact_file}" | render_tsv_table

    echo
    echo "Failed Jobs:"
    colorize_rows_by_status_column 5 < "${failed_jobs_compact_file}" | render_tsv_table

    rm -f "${dataset_summary_file}" "${state_counts_file}" "${status_counts_file}" "${job_overview_file}" "${recent_runs_file}" "${dataset_overview_min_file:-}" "${dataset_tx_rows_file:-}" "${done_jobs_file:-}" "${done_jobs_compact_file:-}" "${failed_jobs_file:-}" "${failed_jobs_compact_file:-}"
    return 0
  fi

  printf "%b\n" "${C_CYAN}Aggregate Benchmark Dashboard${C_RESET} | $(timestamp)"
  printf "Root: %s\n" "${ROOT}"
  printf "Snapshot: %s\n" "${snapshot}"
  printf "Datasets: %d  Jobs: %d\n" "${dataset_count}" "${total_jobs}"
  printf "Progress: %s\n" "$(draw_progress_bar "${processed}" "${total_jobs}")"
  echo
  printf "%b\n" "${C_BLUE}running=${running_count}${C_RESET}  ${C_YELLOW}pending=${pending_count}${C_RESET}  ${C_RED}failed=${failed_count}${C_RESET}  ${C_GREEN}done=${done_count}${C_RESET}  ${C_CYAN}partial=${partial_count}${C_RESET}"

  if [[ "${#DASH_WARNINGS[@]}" -gt 0 ]]; then
    echo
    printf "%b\n" "${C_YELLOW}Warnings:${C_RESET}"
    for warning in "${DASH_WARNINGS[@]}"; do
      printf -- "- %s\n" "${warning}"
    done
  fi

  echo
  echo "State Counts:"
  colorize_rows_by_state_column 1 < "${state_counts_file}" | render_tsv_table

  echo
  echo "Status Counts:"
  colorize_rows_by_status_column 1 < "${status_counts_file}" | render_tsv_table

  echo
  echo "Dataset Summary:"
  render_tsv_table < "${dataset_summary_file}"

  echo
  echo "All Jobs:"
  colorize_rows_by_state_column 5 < "${job_overview_file}" | render_tsv_table

  if [[ "$(awk 'END { print NR-1 }' "${recent_runs_file}")" -gt 0 ]]; then
    echo
    if [[ "${RECENT_LIMIT}" == "0" ]]; then
      echo "All Parsed Runs:"
    else
      echo "Recent Runs (Last ${RECENT_LIMIT}):"
    fi
    render_tsv_table < "${recent_runs_file}"
  fi

  rm -f "${dataset_summary_file}" "${state_counts_file}" "${status_counts_file}" "${job_overview_file}" "${recent_runs_file}" "${dataset_overview_min_file:-}" "${dataset_tx_rows_file:-}" "${done_jobs_file:-}" "${done_jobs_compact_file:-}" "${failed_jobs_file:-}" "${failed_jobs_compact_file:-}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_value "$1" "${2-}"
      ROOT="$2"
      shift 2
      ;;
    --out-tsv)
      require_value "$1" "${2-}"
      OUT_TSV="$2"
      shift 2
      ;;
    --no-refresh)
      REFRESH=0
      shift
      ;;
    --watch)
      if [[ $# -ge 2 ]] && [[ ! "${2-}" =~ ^- ]]; then
        WATCH_SEC="$2"
        shift 2
      else
        WATCH_SEC=20
        shift
      fi
      ;;
    --recent-limit)
      require_value "$1" "${2-}"
      RECENT_LIMIT="$2"
      shift 2
      ;;
    --minimal)
      MINIMAL_MODE=1
      shift
      ;;
    --no-color)
      NO_COLOR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

normalize_root_layout

if [[ -z "${OUT_TSV}" ]]; then
  OUT_TSV="${ROOT}/summaries/aggregate_status_snapshot.tsv"
fi

if ! [[ "${WATCH_SEC}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --watch must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${RECENT_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --recent-limit must be a non-negative integer." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "${OUT_TSV}")"

collect_dataset_dirs
configure_colors

if [[ "${WATCH_SEC}" -gt 0 ]]; then
  while true; do
    if [[ -t 1 ]]; then
      clear
    fi
    render_once
    sleep "${WATCH_SEC}"
  done
else
  render_once
fi
