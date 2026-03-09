"""
Model weight downloader — downloads model weights and databases from various sources.

Supports HuggingFace Hub, direct URL, and S3 with progress bars and resume support.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ct.cloud.weight_downloader")

DEFAULT_CACHE_DIR = Path.home() / ".ct" / "models"


def get_cache_dir() -> Path:
    """Get the model weight cache directory."""
    try:
        from ct.agent.config import Config
        cfg = Config.load()
        custom = cfg.get("data.model_cache")
        if custom:
            return Path(custom)
    except Exception:
        pass
    return DEFAULT_CACHE_DIR


def _cache_path_for_model(model_config: dict) -> Path:
    """Get local cache path for a model."""
    cache_dir = get_cache_dir()
    name = model_config.get("name", "unknown")
    # Sanitize name for filesystem
    safe_name = name.replace("/", "_").replace("\\", "_")
    return cache_dir / safe_name


def is_cached(model_config: dict) -> bool:
    """Check if a model's weights are already cached."""
    path = _cache_path_for_model(model_config)
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


def download_huggingface(model_name: str, cache_path: Path, resume: bool = True) -> None:
    """Download model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(cache_path),
        resume_download=resume,
    )


def download_url(url: str, cache_path: Path, resume: bool = True) -> None:
    """Download from direct URL with progress and resume support."""
    import httpx

    cache_path.mkdir(parents=True, exist_ok=True)

    # Determine output filename from URL
    filename = url.split("/")[-1].split("?")[0]
    output_file = cache_path / filename

    headers = {}
    existing_size = 0
    if resume and output_file.exists():
        existing_size = output_file.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"

    mode = "ab" if existing_size > 0 else "wb"

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=30) as response:
        if response.status_code == 416:
            # Range not satisfiable — file is complete
            return

        total = None
        content_length = response.headers.get("content-length")
        if content_length:
            total = int(content_length) + existing_size

        downloaded = existing_size
        with open(output_file, mode) as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = (downloaded / total) * 100
                    print(f"\r  Downloading {filename}: {pct:.1f}% ({downloaded}/{total} bytes)", end="", flush=True)

    if total:
        print()  # newline after progress


def download_s3(s3_uri: str, cache_path: Path) -> None:
    """Download from S3 bucket."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed. Run: pip install boto3")

    cache_path.mkdir(parents=True, exist_ok=True)

    # Parse s3://bucket/key
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    filename = key.split("/")[-1]

    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(cache_path / filename))


def download_model(model_config: dict, force: bool = False) -> Path:
    """Download a single model/database based on its manifest config.

    Args:
        model_config: Model configuration dict from manifest.
        force: If True, re-download even if cached.

    Returns:
        Path to the cached model directory.
    """
    cache_path = _cache_path_for_model(model_config)
    name = model_config.get("name", "unknown")
    source = model_config.get("source", "huggingface")
    size_gb = model_config.get("size_gb", 0)

    if not force and is_cached(model_config):
        print(f"  Already cached: {name} ({cache_path})")
        return cache_path

    print(f"  Downloading {name} ({size_gb}GB) from {source}...")

    if source == "huggingface":
        download_huggingface(name, cache_path)
    elif source == "url":
        url = model_config.get("url", "")
        if not url:
            raise ValueError(f"No URL specified for {name}")
        download_url(url, cache_path)
    elif source == "s3":
        s3_uri = model_config.get("url", "")
        if not s3_uri:
            raise ValueError(f"No S3 URI specified for {name}")
        download_s3(s3_uri, cache_path)
    else:
        raise ValueError(f"Unknown download source: {source}")

    print(f"  Downloaded: {name} -> {cache_path}")
    return cache_path


def pull_tool_weights(tool_name: str, include_optional: bool = False, force: bool = False) -> dict:
    """Download all model weights and databases for a tool.

    Args:
        tool_name: Tool name (e.g., 'structure.esmfold').
        include_optional: Whether to download optional databases.
        force: If True, re-download even if cached.

    Returns:
        Summary dict.
    """
    from ct.cloud.manifest import get_tool_config

    config = get_tool_config(tool_name)
    if not config:
        return {"summary": f"Tool '{tool_name}' not found in manifest.", "error": "not_found"}

    display_name = config.get("display_name", tool_name)
    models = config.get("models", [])
    databases = config.get("databases", [])

    downloaded = []
    skipped = []

    # Download required models
    for model in models:
        if model.get("required", True):
            try:
                path = download_model(model, force=force)
                downloaded.append(model["name"])
            except Exception as e:
                return {"summary": f"Failed to download {model['name']}: {e}", "error": str(e)}

    # Download databases
    for db in databases:
        if db.get("optional", False) and not include_optional:
            size = db.get("size_gb", 0)
            skipped.append(f"{db['name']} ({size}GB, optional)")
            continue
        try:
            path = download_model(db, force=force)
            downloaded.append(db["name"])
        except Exception as e:
            return {"summary": f"Failed to download {db['name']}: {e}", "error": str(e)}

    summary_parts = [f"Pulled weights for {display_name}."]
    if downloaded:
        summary_parts.append(f"Downloaded: {', '.join(downloaded)}.")
    if skipped:
        summary_parts.append(f"Skipped optional: {', '.join(skipped)}.")

    return {
        "summary": " ".join(summary_parts),
        "downloaded": downloaded,
        "skipped": skipped,
    }
