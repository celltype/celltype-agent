"""Local MMseqs2 MSA search implementation optimized for high-throughput runs."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

UNIREF_DB = "uniref30_2302_db"
ENV_DB = "colabfold_envdb_202108_db"
DEFAULT_DB_PATHS = [
    "/vol/colabfold_db",
    "/home/ubuntu/.cache/colabfold_db",
    os.path.expanduser("~/.cache/colabfold_db"),
]
SEARCH_RESULT_NAME = "bfd.mgnify30.metaeuk30.smag30.a3m"
ALLOWED_BACKENDS = {"auto", "cli", "server"}
ALLOWED_SERVER_SEARCH_TYPES = {"colabfold"}


def _default_backend() -> str:
    override = str(os.environ.get("MSA_SEARCH_BACKEND", "")).strip().lower()
    if override in ALLOWED_BACKENDS:
        return override
    return "server" if str(os.environ.get("MSA_SEARCH_SERVER_URL", "")).strip() else "cli"


def _normalized_server_url(server_url: str = "") -> str:
    return str(server_url or os.environ.get("MSA_SEARCH_SERVER_URL", "")).strip().rstrip("/")


def _normalized_server_search_type(server_search_type: str = "") -> str:
    value = str(server_search_type or os.environ.get("MSA_SEARCH_SERVER_SEARCH_TYPE", "colabfold")).strip().lower()
    if not value:
        value = "colabfold"
    if value not in ALLOWED_SERVER_SEARCH_TYPES:
        raise ValueError(
            f"Unsupported MSA server search type `{value}`. Expected one of {sorted(ALLOWED_SERVER_SEARCH_TYPES)}."
        )
    return value


def _normalized_server_timeout_s(server_timeout_s: int | None = None) -> int:
    if server_timeout_s is None:
        raw = str(os.environ.get("MSA_SEARCH_SERVER_TIMEOUT_S", "120")).strip()
        server_timeout_s = int(raw or "120")
    return max(1, int(server_timeout_s))


def _resolve_backend(backend: str = "", server_url: str = "") -> tuple[str, str]:
    requested = str(backend or _default_backend()).strip().lower() or _default_backend()
    if requested not in ALLOWED_BACKENDS:
        raise ValueError(f"Unsupported MSA backend `{requested}`. Expected one of {sorted(ALLOWED_BACKENDS)}.")

    resolved = requested
    if requested == "auto":
        resolved = "server" if _normalized_server_url(server_url) else "cli"
    if resolved == "server" and not _normalized_server_url(server_url):
        raise ValueError("MSA server backend requires `server_url` or MSA_SEARCH_SERVER_URL.")
    return requested, resolved


@lru_cache(maxsize=1)
def _find_db_base() -> Path:
    explicit = os.environ.get("COLABFOLD_DB")
    if explicit:
        path = Path(explicit)
        if path.joinpath(f"{UNIREF_DB}.dbtype").exists():
            return path
        raise FileNotFoundError(f"COLABFOLD_DB={explicit} does not contain {UNIREF_DB}.dbtype")

    for candidate in DEFAULT_DB_PATHS:
        path = Path(candidate)
        if path.joinpath(f"{UNIREF_DB}.dbtype").exists():
            return path

    raise FileNotFoundError(
        "ColabFold databases not found. Set COLABFOLD_DB env var or place databases in "
        + " or ".join(DEFAULT_DB_PATHS)
    )


@lru_cache(maxsize=1)
def _find_mmseqs() -> str:
    for candidate in ["mmseqs", "/usr/local/bin/mmseqs", "/opt/mmseqs/bin/mmseqs"]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("mmseqs binary not found in PATH")


@lru_cache(maxsize=1)
def _system_ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / 1024**3
    except (AttributeError, OSError, ValueError):
        return 0.0


def _has_db_index(dbbase: Path, db_name: str) -> bool:
    return any(
        dbbase.joinpath(candidate).exists()
        for candidate in (
            f"{db_name}.index",
            f"{db_name}_seq.index",
            f"{db_name}_aln.index",
        )
    )


def _gpu_ready(dbbase: Path, db_name: str) -> bool:
    return dbbase.joinpath(f"{db_name}.GPU_READY").exists()


def _parse_database_selection(database: str | list[str]) -> tuple[list[str], str]:
    if isinstance(database, str):
        raw_values = [database]
    elif isinstance(database, list):
        raw_values = database
    else:
        raw_values = ["all"]

    normalized: list[str] = []
    for item in raw_values:
        value = str(item).strip().lower()
        if not value:
            continue
        if value == "all":
            return ["all"], "all"
        if value in {"uniref30_2302", UNIREF_DB, "uniref30"}:
            canonical = "uniref30_2302"
        elif value in {"colabfold_envdb_202108", ENV_DB, "colabfold_envdb", "envdb"}:
            canonical = "colabfold_envdb_202108"
        else:
            raise ValueError(f"Unsupported database `{value}`.")
        if canonical not in normalized:
            normalized.append(canonical)

    if not normalized:
        normalized = ["all"]
    return normalized, "+".join(normalized)


def _should_use_envdb(databases: list[str]) -> bool:
    return "all" in databases or "colabfold_envdb_202108" in databases


def _sequence_cache_key(sequence: str, db_key: str) -> str:
    return hashlib.sha256(f"{db_key}\n{sequence}".encode("utf-8")).hexdigest()


def _colabfold_server_mode(normalized_databases: list[str], server_search_type: str) -> str:
    if server_search_type != "colabfold":
        raise ValueError(
            "Open-source ColabFold server only supports `server_search_type=colabfold`."
        )
    return "env" if _should_use_envdb(normalized_databases) else "all"


def _colabfold_server_prefix(server_url: str) -> str:
    base = _normalized_server_url(server_url)
    if not base:
        raise ValueError("MSA server backend requires `server_url` or MSA_SEARCH_SERVER_URL.")
    return base if base.endswith("/api") else f"{base}/api"


def _colabfold_server_request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    timeout_s: int,
) -> bytes:
    request = urllib_request.Request(url, data=data, method=method)
    with urllib_request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _colabfold_server_submit(
    sequence: str,
    *,
    server_url: str,
    server_search_type: str,
    normalized_databases: list[str],
    timeout_s: int,
) -> dict:
    mode = _colabfold_server_mode(normalized_databases, server_search_type)
    payload = urllib_parse.urlencode({"q": f">101\n{sequence}\n", "mode": mode}).encode("utf-8")
    response = _colabfold_server_request(
        "POST",
        f"{_colabfold_server_prefix(server_url)}/ticket/msa",
        data=payload,
        timeout_s=timeout_s,
    )
    return json.loads(response.decode("utf-8"))


def _colabfold_server_status(server_url: str, ticket_id: str, timeout_s: int) -> dict:
    response = _colabfold_server_request(
        "GET",
        f"{_colabfold_server_prefix(server_url)}/ticket/{ticket_id}",
        timeout_s=timeout_s,
    )
    return json.loads(response.decode("utf-8"))


def _colabfold_server_download_result(
    server_url: str,
    ticket_id: str,
    *,
    timeout_s: int,
    max_attempts: int = 3,
    retry_delay_s: float = 1.0,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _colabfold_server_request(
                "GET",
                f"{_colabfold_server_prefix(server_url)}/result/download/{ticket_id}",
                timeout_s=timeout_s,
            )
        except urllib_error.HTTPError as exc:
            last_error = exc
            if exc.code != 404 or attempt == max_attempts - 1:
                break
        except urllib_error.URLError as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
        time.sleep(retry_delay_s)

    local_root = str(os.environ.get("MSA_SEARCH_SERVER_JOB_ROOT", "")).strip()
    if local_root:
        local_tar = Path(local_root) / ticket_id / f"mmseqs_results_{ticket_id}.tar.gz"
        if local_tar.is_file():
            return local_tar.read_bytes()

    if last_error is not None:
        raise last_error
    raise ValueError(f"ColabFold server result for `{ticket_id}` could not be downloaded.")


def _colabfold_server_extract_alignment(
    tar_bytes: bytes,
    *,
    use_env: bool,
    query_id: int = 101,
) -> tuple[str, str]:
    a3m_files = ["uniref.a3m"]
    if use_env:
        a3m_files.append(SEARCH_RESULT_NAME)

    a3m_lines: dict[int, list[str]] = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar_gz:
        for name in a3m_files:
            try:
                member = tar_gz.getmember(name)
            except KeyError:
                continue
            extracted = tar_gz.extractfile(member)
            if extracted is None:
                continue

            update_query_id = True
            current_query_id: int | None = None
            for raw_line in extracted.read().decode("utf-8").splitlines(keepends=True):
                if "\x00" in raw_line:
                    raw_line = raw_line.replace("\x00", "")
                    update_query_id = True
                if raw_line.startswith(">") and update_query_id:
                    current_query_id = int(raw_line[1:].strip())
                    update_query_id = False
                    a3m_lines.setdefault(current_query_id, [])
                if current_query_id is not None:
                    a3m_lines[current_query_id].append(raw_line)

    if query_id not in a3m_lines:
        raise ValueError("ColabFold server result did not contain the requested query alignment.")

    return "".join(a3m_lines[query_id]), "colabfold-ticket"


def _load_cached_a3m(cache_dir: str, sequence: str, db_key: str) -> str:
    if not cache_dir:
        return ""
    path = Path(cache_dir) / f"{_sequence_cache_key(sequence, db_key)}.a3m"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").replace("\x00", "")


def _save_cached_a3m(cache_dir: str, sequence: str, db_key: str, msa_text: str) -> None:
    if not cache_dir:
        return
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_sequence_cache_key(sequence, db_key)}.a3m"
    path.write_text(msa_text, encoding="utf-8")


def _existing_mmseqs_output(path: str | Path | None) -> bool:
    if not path:
        return False
    path_obj = Path(path)
    return path_obj.exists() or path_obj.with_suffix(".dbtype").exists()


def _run_mmseqs(mmseqs: str, params: list, check_exists: str | Path | None = None) -> None:
    if _existing_mmseqs_output(check_exists):
        logger.info("Skipping %s: %s already exists", params[0], check_exists)
        return

    os.environ["MMSEQS_FORCE_MERGE"] = "1"
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    cmd = [mmseqs] + [str(part) for part in params]
    logger.info("Running mmseqs: %s", " ".join(cmd[:8]))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-4000:]
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=stderr_tail,
        )


def _fetch_alignment_from_server(
    sequence: str,
    normalized_databases: list[str],
    *,
    e_value: float,
    iterations: int,
    server_url: str,
    server_search_type: str,
    server_timeout_s: int,
) -> dict:
    t0 = time.time()
    submission = _colabfold_server_submit(
        sequence,
        server_url=server_url,
        server_search_type=server_search_type,
        normalized_databases=normalized_databases,
        timeout_s=server_timeout_s,
    )
    ticket_id = str(submission.get("id", "")).strip()
    if not ticket_id:
        raise ValueError(f"ColabFold server did not return a ticket id: {submission}")

    status = submission
    while status.get("status") in {"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"}:
        time.sleep(2)
        status = _colabfold_server_status(server_url, ticket_id, server_timeout_s)

    terminal_status = str(status.get("status", "")).strip().upper()
    tar_bytes = _colabfold_server_download_result(
        server_url,
        ticket_id,
        timeout_s=server_timeout_s,
    )
    t_total = time.time() - t0

    msa_content, alignment_key = _colabfold_server_extract_alignment(
        tar_bytes,
        use_env=_should_use_envdb(normalized_databases),
    )
    if not msa_content.strip():
        raise ValueError("ColabFold server returned an empty A3M alignment.")
    if terminal_status not in {"COMPLETE", "ERROR"}:
        raise ValueError(
            f"ColabFold server ticket `{ticket_id}` ended in unexpected status `{status.get('status')}`."
        )

    return {
        "msa": msa_content,
        "num_sequences": msa_content.count(">"),
        "database": normalized_databases,
        "query_length": len(sequence),
        "msa_size_bytes": len(msa_content),
        "metrics": {
            "vram_before_mb": 0,
            "vram_peak_mb": 0,
            "time_createdb_s": 0.0,
            "time_search_s": round(t_total, 2),
            "time_total_s": round(t_total, 2),
            "hardware": "Open-source ColabFold/MMseqs server",
            "cache_hit": False,
            "cache_level": "miss",
            "db_load_mode": None,
            "threads": 0,
            "use_gpu": False,
            "backend": "server",
            "server_url": _normalized_server_url(server_url),
            "server_alignment_key": alignment_key,
            "server_status": terminal_status.lower(),
        },
    }


def _default_db_load_mode(dbbase: Path) -> int:
    override = os.environ.get("MSA_SEARCH_DB_LOAD_MODE", "").strip()
    if override:
        return int(override)

    if _has_db_index(dbbase, UNIREF_DB) and _system_ram_gb() >= 256:
        return 3
    if _has_db_index(dbbase, UNIREF_DB):
        return 2
    return 0


def _default_threads() -> int:
    override = os.environ.get("MSA_SEARCH_THREADS", "").strip()
    if override:
        return max(1, int(override))
    return max(1, min(os.cpu_count() or 8, 128))


def _default_use_gpu(dbbase: Path) -> bool:
    override = os.environ.get("MSA_SEARCH_USE_GPU", "").strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    if override in {"0", "false", "no"}:
        return False
    if not shutil.which("nvidia-smi"):
        return False
    return _gpu_ready(dbbase, UNIREF_DB) and _gpu_ready(dbbase, ENV_DB)


def _normalize_sequence(sequence: str) -> str:
    return sequence.strip().upper().replace(" ", "").replace("\n", "")


def _search_monomer(
    mmseqs: str,
    dbbase: Path,
    workdir: Path,
    *,
    use_env: bool,
    threads: int,
    db_load_mode: int,
    prefilter_mode: int,
    sensitivity: float,
    num_iterations: int,
    use_gpu: bool,
    e_value: float,
) -> None:
    search_gpu_args = ["--gpu", "1"] if use_gpu else []
    search_param = [
        "--num-iterations", str(max(3, num_iterations)),
        "--db-load-mode", str(db_load_mode),
        "-a",
        "-e", f"{max(e_value, 0.1):.4g}",
        "--max-seqs", "10000",
        "--prefilter-mode", str(prefilter_mode),
        "-s", f"{sensitivity:.1f}",
    ] + search_gpu_args
    filter_param = [
        "--filter-msa", "1",
        "--filter-min-enable", "1000",
        "--diff", "3000",
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", "0.95",
    ]
    expand_param = [
        "--expansion-mode", "0",
        "-e", "inf",
        "--expand-filter-clusters", "1",
        "--max-seq-id", "0.95",
    ]
    align_eval = 10
    qsc = 0.8
    max_accept = 100000
    seq_suffix = "_seq"
    aln_suffix = "_aln"
    work = workdir
    db = dbbase

    _run_mmseqs(
        mmseqs,
        [
            "search", work / "qdb", db / UNIREF_DB, work / "res", work / "tmp",
            "--threads", str(threads),
        ] + search_param,
        check_exists=work / "uniref.a3m",
    )
    _run_mmseqs(mmseqs, ["mvdb", work / "tmp/latest/profile_1", work / "prof_res"])
    _run_mmseqs(mmseqs, ["lndb", work / "qdb_h", work / "prof_res_h"])
    _run_mmseqs(
        mmseqs,
        [
            "expandaln", work / "qdb", db / f"{UNIREF_DB}{seq_suffix}",
            work / "res", db / f"{UNIREF_DB}{aln_suffix}", work / "res_exp",
            "--db-load-mode", str(db_load_mode), "--threads", str(threads),
        ] + expand_param,
        check_exists=work / "uniref.a3m",
    )
    _run_mmseqs(
        mmseqs,
        [
            "align", work / "prof_res", db / f"{UNIREF_DB}{seq_suffix}",
            work / "res_exp", work / "res_exp_realign",
            "--db-load-mode", str(db_load_mode),
            "-e", str(align_eval), "--max-accept", str(max_accept),
            "--threads", str(threads), "--alt-ali", "10", "-a",
        ],
        check_exists=work / "uniref.a3m",
    )
    _run_mmseqs(
        mmseqs,
        [
            "filterresult", work / "qdb", db / f"{UNIREF_DB}{seq_suffix}",
            work / "res_exp_realign", work / "res_exp_realign_filter",
            "--db-load-mode", str(db_load_mode),
            "--qid", "0", "--qsc", str(qsc), "--diff", "0",
            "--threads", str(threads), "--max-seq-id", "1.0",
            "--filter-min-enable", "100",
        ],
        check_exists=work / "uniref.a3m",
    )
    _run_mmseqs(
        mmseqs,
        [
            "result2msa", work / "qdb", db / f"{UNIREF_DB}{seq_suffix}",
            work / "res_exp_realign_filter", work / "uniref.a3m",
            "--msa-format-mode", "6",
            "--db-load-mode", str(db_load_mode), "--threads", str(threads),
        ] + filter_param,
    )
    for db_name in ["res_exp_realign_filter", "res_exp_realign", "res_exp", "res"]:
        _run_mmseqs(mmseqs, ["rmdb", work / db_name])

    if use_env:
        _run_mmseqs(
            mmseqs,
            [
                "search", work / "prof_res", db / ENV_DB, work / "res_env", work / "tmp3",
                "--threads", str(threads),
            ] + search_param,
            check_exists=work / SEARCH_RESULT_NAME,
        )
        _run_mmseqs(
            mmseqs,
            [
                "expandaln", work / "prof_res", db / f"{ENV_DB}{seq_suffix}",
                work / "res_env", db / f"{ENV_DB}{aln_suffix}", work / "res_env_exp",
                "-e", "inf", "--expansion-mode", "0",
                "--db-load-mode", str(db_load_mode), "--threads", str(threads),
            ],
            check_exists=work / SEARCH_RESULT_NAME,
        )
        _run_mmseqs(
            mmseqs,
            [
                "align", work / "tmp3/latest/profile_1", db / f"{ENV_DB}{seq_suffix}",
                work / "res_env_exp", work / "res_env_exp_realign",
                "--db-load-mode", str(db_load_mode),
                "-e", str(align_eval), "--max-accept", str(max_accept),
                "--threads", str(threads), "--alt-ali", "10", "-a",
            ],
            check_exists=work / SEARCH_RESULT_NAME,
        )
        _run_mmseqs(
            mmseqs,
            [
                "filterresult", work / "qdb", db / f"{ENV_DB}{seq_suffix}",
                work / "res_env_exp_realign", work / "res_env_exp_realign_filter",
                "--db-load-mode", str(db_load_mode),
                "--qid", "0", "--qsc", str(qsc), "--diff", "0",
                "--max-seq-id", "1.0", "--threads", str(threads),
                "--filter-min-enable", "100",
            ],
            check_exists=work / SEARCH_RESULT_NAME,
        )
        _run_mmseqs(
            mmseqs,
            [
                "result2msa", work / "qdb", db / f"{ENV_DB}{seq_suffix}",
                work / "res_env_exp_realign_filter", work / SEARCH_RESULT_NAME,
                "--msa-format-mode", "6",
                "--db-load-mode", str(db_load_mode), "--threads", str(threads),
            ] + filter_param,
        )
        for db_name in ["res_env_exp_realign_filter", "res_env_exp_realign", "res_env_exp", "res_env"]:
            _run_mmseqs(mmseqs, ["rmdb", work / db_name])

    if use_env:
        _run_mmseqs(mmseqs, ["mergedbs", work / "qdb", work / "final.a3m", work / "uniref.a3m", work / SEARCH_RESULT_NAME])
        _run_mmseqs(mmseqs, ["rmdb", work / SEARCH_RESULT_NAME])
        _run_mmseqs(mmseqs, ["rmdb", work / "uniref.a3m"])
    else:
        _run_mmseqs(mmseqs, ["mvdb", work / "uniref.a3m", work / "final.a3m"])

    _run_mmseqs(
        mmseqs,
        ["unpackdb", work / "final.a3m", work / ".", "--unpack-name-mode", "0", "--unpack-suffix", ".a3m"],
    )
    _run_mmseqs(mmseqs, ["rmdb", work / "final.a3m"])
    _run_mmseqs(mmseqs, ["rmdb", work / "prof_res"])
    _run_mmseqs(mmseqs, ["rmdb", work / "prof_res_h"])
    for tmp_dir in ["tmp", "tmp3"]:
        path = work / tmp_dir
        if path.exists():
            shutil.rmtree(path)


def _search_with_local_mmseqs(
    clean_seq: str,
    normalized_databases: list[str],
    *,
    e_value: float,
    iterations: int,
    cache_dir: str,
    in_memory_cache: dict[str, str] | None,
    cache_stats: dict[str, int] | None,
    threads: int | None,
    use_gpu: bool | None,
    db_load_mode: int | None,
    prefilter_mode: int,
    sensitivity: float,
) -> dict:
    mmseqs = _find_mmseqs()
    dbbase = _find_db_base()
    use_env = _should_use_envdb(normalized_databases)
    chosen_threads = threads if threads is not None else _default_threads()
    chosen_db_load_mode = db_load_mode if db_load_mode is not None else _default_db_load_mode(dbbase)
    chosen_use_gpu = use_gpu if use_gpu is not None else _default_use_gpu(dbbase)

    t0 = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="msa_search_"))
    try:
        query_fasta = workdir / "query.fas"
        query_fasta.write_text(f">101\n{clean_seq}\n")

        t_createdb = time.time()
        _run_mmseqs(mmseqs, ["createdb", query_fasta, workdir / "qdb", "--shuffle", "0", "--dbtype", "1"])
        t_createdb = time.time() - t_createdb

        (workdir / "qdb.lookup").write_text("0\t101\t0\n")

        t_search = time.time()
        _search_monomer(
            mmseqs=mmseqs,
            dbbase=dbbase,
            workdir=workdir,
            use_env=use_env,
            threads=chosen_threads,
            db_load_mode=chosen_db_load_mode,
            prefilter_mode=prefilter_mode,
            sensitivity=sensitivity,
            num_iterations=max(1, int(iterations)),
            use_gpu=bool(chosen_use_gpu),
            e_value=float(e_value),
        )
        t_search = time.time() - t_search

        a3m_file = workdir / "0.a3m"
        if not a3m_file.exists():
            raise RuntimeError("MMseqs2 search produced no alignment output.")

        msa_content = a3m_file.read_text(encoding="utf-8").replace("\x00", "")
        return {
            "msa": msa_content,
            "num_sequences": msa_content.count(">"),
            "database": normalized_databases,
            "query_length": len(clean_seq),
            "msa_size_bytes": len(msa_content),
            "metrics": {
                "vram_before_mb": 0,
                "vram_peak_mb": 0,
                "time_createdb_s": round(t_createdb, 2),
                "time_search_s": round(t_search, 2),
                "time_total_s": round(time.time() - t0, 2),
                "hardware": f"Local MMseqs2 ({'GPU' if chosen_use_gpu else 'CPU'}, {chosen_threads} threads)",
                "cache_hit": False,
                "cache_level": "miss",
                "db_load_mode": chosen_db_load_mode,
                "threads": chosen_threads,
                "use_gpu": bool(chosen_use_gpu),
                "backend": "cli",
            },
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def get_alignment_for_sequence(
    sequence: str,
    database: str | list[str] = "all",
    *,
    e_value: float = 0.0001,
    iterations: int = 1,
    cache_dir: str = "",
    in_memory_cache: dict[str, str] | None = None,
    cache_stats: dict[str, int] | None = None,
    threads: int | None = None,
    use_gpu: bool | None = None,
    db_load_mode: int | None = None,
    prefilter_mode: int = 1,
    sensitivity: float = 8.0,
    backend: str = "",
    server_url: str = "",
    server_search_type: str = "",
    server_timeout_s: int | None = None,
) -> dict:
    clean_seq = _normalize_sequence(sequence)
    if not clean_seq:
        raise ValueError("No sequence provided.")

    normalized_databases, database_key = _parse_database_selection(database)
    cache_dir = str(cache_dir or os.environ.get("MSA_SEARCH_CACHE_DIR", "")).strip()
    memory_key = f"{database_key}:{clean_seq}"

    if in_memory_cache is not None and memory_key in in_memory_cache:
        if cache_stats is not None:
            cache_stats["memory_hits"] = cache_stats.get("memory_hits", 0) + 1
        cached_msa = in_memory_cache[memory_key]
        return {
            "msa": cached_msa,
            "num_sequences": cached_msa.count(">"),
            "database": normalized_databases,
            "query_length": len(clean_seq),
            "msa_size_bytes": len(cached_msa),
            "metrics": {
                "vram_before_mb": 0,
                "vram_peak_mb": 0,
                "time_createdb_s": 0.0,
                "time_search_s": 0.0,
                "time_total_s": 0.0,
                "hardware": "Local MMseqs2 memory cache",
                "cache_hit": True,
                "cache_level": "memory",
                "db_load_mode": None,
                "threads": 0,
                "use_gpu": False,
                "backend": "cache",
            },
        }

    cached_msa = _load_cached_a3m(cache_dir, clean_seq, database_key)
    if cached_msa:
        if in_memory_cache is not None:
            in_memory_cache[memory_key] = cached_msa
        if cache_stats is not None:
            cache_stats["disk_hits"] = cache_stats.get("disk_hits", 0) + 1
        return {
            "msa": cached_msa,
            "num_sequences": cached_msa.count(">"),
            "database": normalized_databases,
            "query_length": len(clean_seq),
            "msa_size_bytes": len(cached_msa),
            "metrics": {
                "vram_before_mb": 0,
                "vram_peak_mb": 0,
                "time_createdb_s": 0.0,
                "time_search_s": 0.0,
                "time_total_s": 0.0,
                "hardware": "Local MMseqs2 disk cache",
                "cache_hit": True,
                "cache_level": "disk",
                "db_load_mode": None,
                "threads": 0,
                "use_gpu": False,
                "backend": "cache",
            },
        }

    requested_backend, resolved_backend = _resolve_backend(backend, server_url)
    if resolved_backend == "server":
        try:
            result = _fetch_alignment_from_server(
                clean_seq,
                normalized_databases,
                e_value=e_value,
                iterations=iterations,
                server_url=server_url,
                server_search_type=_normalized_server_search_type(server_search_type),
                server_timeout_s=_normalized_server_timeout_s(server_timeout_s),
            )
        except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if requested_backend != "auto":
                raise
            logger.warning("MSA server backend unavailable, falling back to CLI MMseqs: %s", exc)
            resolved_backend = "cli"

    if resolved_backend == "cli":
        result = _search_with_local_mmseqs(
            clean_seq,
            normalized_databases,
            e_value=e_value,
            iterations=iterations,
            cache_dir=cache_dir,
            in_memory_cache=in_memory_cache,
            cache_stats=cache_stats,
            threads=threads,
            use_gpu=use_gpu,
            db_load_mode=db_load_mode,
            prefilter_mode=prefilter_mode,
            sensitivity=sensitivity,
        )

    msa_content = result["msa"]
    _save_cached_a3m(cache_dir, clean_seq, database_key, msa_content)
    if in_memory_cache is not None:
        in_memory_cache[memory_key] = msa_content
    if cache_stats is not None:
        cache_stats["misses"] = cache_stats.get("misses", 0) + 1
    return result


def cache_alignment_for_sequence(
    sequence: str,
    msa: str,
    database: str | list[str] = "all",
    *,
    cache_dir: str = "",
    in_memory_cache: dict[str, str] | None = None,
) -> dict:
    clean_seq = _normalize_sequence(sequence)
    if not clean_seq:
        raise ValueError("No sequence provided.")

    msa_text = str(msa or "")
    if not msa_text.strip():
        raise ValueError("No MSA provided.")

    normalized_databases, database_key = _parse_database_selection(database)
    cache_dir = str(cache_dir or os.environ.get("MSA_SEARCH_CACHE_DIR", "")).strip()
    memory_key = f"{database_key}:{clean_seq}"
    if in_memory_cache is not None:
        in_memory_cache[memory_key] = msa_text
    _save_cached_a3m(cache_dir, clean_seq, database_key, msa_text)
    return {
        "msa": msa_text,
        "num_sequences": msa_text.count(">"),
        "database": normalized_databases,
        "query_length": len(clean_seq),
        "msa_size_bytes": len(msa_text),
        "metrics": {
            "cache_hit": True,
            "cache_level": "seeded",
        },
    }


def run(
    sequence: str = "",
    database: str | list[str] = "all",
    e_value: float = 0.0001,
    iterations: int = 1,
    session_id: str = "",
    cache_dir: str = "",
    threads: int | None = None,
    use_gpu: bool | None = None,
    db_load_mode: int | None = None,
    prefilter_mode: int = 1,
    sensitivity: float = 8.0,
    backend: str = "",
    server_url: str = "",
    server_search_type: str = "",
    server_timeout_s: int | None = None,
    **kwargs,
) -> dict:
    """Run a local MMseqs2 MSA search using on-disk ColabFold databases only."""
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    try:
        result = get_alignment_for_sequence(
            sequence,
            database=database,
            e_value=e_value,
            iterations=iterations,
            cache_dir=cache_dir,
            threads=threads,
            use_gpu=use_gpu,
            db_load_mode=db_load_mode,
            prefilter_mode=prefilter_mode,
            sensitivity=sensitivity,
            backend=backend,
            server_url=server_url,
            server_search_type=server_search_type,
            server_timeout_s=server_timeout_s,
        )
    except (FileNotFoundError, ValueError, urllib_error.URLError, TimeoutError) as exc:
        return {"summary": f"Error: Local MSA setup failed: {exc}", "error": "local_msa_unavailable"}
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "summary": f"Error: MMseqs2 command failed: {exc.cmd[1] if len(exc.cmd) > 1 else 'unknown'} — {stderr[:500]}",
            "error": "mmseqs_error",
            "detail": stderr[:2000],
        }
    except Exception as exc:
        return {"summary": f"Error: Local MSA search failed: {exc}", "error": str(exc)}

    if session_id:
        ws_dir = Path(f"/vol/workspace/{session_id}")
        ws_dir.mkdir(parents=True, exist_ok=True)
        (ws_dir / "msa.a3m").write_text(result["msa"])

    if result["metrics"]["cache_hit"]:
        summary = (
            f"MSA cache hit: reused {result['num_sequences']} homologous sequences for "
            f"{result['query_length']}-residue query."
        )
    else:
        if result["metrics"].get("backend") == "server":
            summary = (
                f"MSA server search completed: found {result['num_sequences']} homologous sequences for "
                f"{result['query_length']}-residue query."
            )
        else:
            use_env = _should_use_envdb(result["database"])
            summary = (
                f"Local MSA search completed: found {result['num_sequences']} homologous sequences for "
                f"{result['query_length']}-residue query."
                + (" Searched UniRef30 + ColabFold EnvDB." if use_env else " Searched UniRef30 only.")
            )

    return {
        "summary": summary,
        **result,
    }
