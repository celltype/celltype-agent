"""ColabFold MSA search implementation.

Uses the public ColabFold MMseqs2 API server for homology search.
No local database needed — queries https://api.colabfold.com.
Hardware: CPU-only.
"""

import io
import tarfile
import time


def run(sequence: str = "", database: str = "colabfold_envdb", session_id: str = "", **kwargs) -> dict:
    """Search for homologous sequences using ColabFold MSA search API."""
    if not sequence:
        return {"summary": "Error: No sequence provided.", "error": "no_sequence"}

    t0 = time.time()
    clean_seq = sequence.strip().upper().replace(" ", "").replace("\n", "")
    seq_len = len(clean_seq)

    try:
        import requests
    except ImportError:
        return {"summary": "Error: requests library not installed.", "error": "missing_dep"}

    api_url = "https://api.colabfold.com"
    mode = "env" if "env" in database else "all"

    try:
        # Step 1: Submit MSA search job
        query = f">101\n{clean_seq}\n"
        t_submit = time.time()
        response = requests.post(
            f"{api_url}/ticket/msa",
            data={"q": query, "mode": mode},
            timeout=30,
        )
        response.raise_for_status()
        ticket = response.json()
        ticket_id = ticket.get("id", "")
        t_submit = time.time() - t_submit

        if not ticket_id:
            return {"summary": f"Error: ColabFold API returned no ticket ID. Response: {ticket}", "error": "api_error"}

        # Step 2: Poll for results
        t_search = time.time()
        max_polls = 120
        for _ in range(max_polls):
            status_resp = requests.get(f"{api_url}/ticket/{ticket_id}", timeout=10)
            status = status_resp.json()
            job_status = status.get("status", "")

            if job_status == "COMPLETE":
                break
            elif job_status == "ERROR":
                return {
                    "summary": f"Error: ColabFold MSA search failed: {status.get('error', 'unknown')}",
                    "error": "search_failed",
                }
            elif job_status in ("UNKNOWN", "RUNNING", "PENDING"):
                time.sleep(5)
            else:
                time.sleep(5)
        else:
            return {"summary": "Error: ColabFold MSA search timed out after 10 minutes.", "error": "timeout"}
        t_search = time.time() - t_search

        # Step 3: Download results
        t_download = time.time()
        result_resp = requests.get(f"{api_url}/result/download/{ticket_id}", timeout=60)
        result_resp.raise_for_status()

        msa_content = ""
        tar_data = io.BytesIO(result_resp.content)
        try:
            with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".a3m"):
                        f = tar.extractfile(member)
                        if f:
                            msa_content = f.read().decode("utf-8", errors="ignore")
                            break
        except tarfile.ReadError:
            msa_content = result_resp.text
        t_download = time.time() - t_download

        if not msa_content:
            return {
                "summary": "Error: No A3M alignment found in ColabFold results.",
                "error": "no_alignment",
            }

        num_sequences = msa_content.count(">")

        if session_id:
            import os
            workspace_dir = f"/vol/workspace/{session_id}"
            os.makedirs(workspace_dir, exist_ok=True)
            with open(f"{workspace_dir}/msa.a3m", "w") as f:
                f.write(msa_content)

        msa_preview = msa_content[:3000]
        if len(msa_content) > 3000:
            msa_preview += f"\n... [{len(msa_content)} total characters]"

        return {
            "summary": (
                f"MSA search: found {num_sequences} homologous sequences "
                f"for {seq_len}-residue query in {database}."
            ),
            "msa": msa_preview,
            "num_sequences": num_sequences,
            "database": database,
            "query_length": seq_len,
            "msa_size_bytes": len(msa_content),
            "metrics": {
                "vram_before_mb": 0,
                "vram_peak_mb": 0,
                "time_submit_s": round(t_submit, 2),
                "time_search_s": round(t_search, 2),
                "time_download_s": round(t_download, 2),
                "time_total_s": round(time.time() - t0, 2),
                "hardware": "CPU-only (API-based)",
            },
        }

    except requests.exceptions.ConnectionError:
        return {
            "summary": "Error: Cannot connect to ColabFold API server. Check internet connection.",
            "error": "connection_error",
        }
    except Exception as e:
        return {
            "summary": f"Error: MSA search failed: {e}",
            "error": str(e),
        }
