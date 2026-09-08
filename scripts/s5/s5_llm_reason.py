#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S5 – LLM Reasoner

- Reads S4.2 JSON files that contain an `llm_input` block.
- Converts `llm_input` to YAML and injects it into an external prompt template.
- Calls a configured LLM backend (Azure OpenAI or Ollama).
- Parses the JSON output from the LLM and writes a separate *.llm.json file.

Config:
    - LLM backends are defined in ``configs/llm_backends.yaml``.
    - Secrets are referenced as ``${ENVIRONMENT_VARIABLE}``; they are never
      stored directly in this repository.
      Each backend may include:
        provider: "azure" | "ollama"
        endpoint / api_key / api_version / model / proxy (for azure)
        base_url / model / proxy (for ollama)
        default_params: temperature, top_p, etc.

Prompt:
    - The prompt template file MUST contain the literal placeholder:
        <<<SCENARIO_YAML>>>
      This script will replace that marker with the YAML string.
      We do *not* use Python str.format() on the prompt to avoid KeyError from {…}.

Usage examples:

    # Dry run: only write the prompts that would be sent
    python scripts/s5/s5_llm_reason.py \
        --config configs/llm_backends.yaml \
        --backend ollama-local \
        --prompt-file scripts/s5/prompts/base_prompt.txt \
        --input-dir artifacts/train650_val50/val50/s4/val50 \
        --output-dir artifacts/train650_val50/val50/s5/base_prompt \
        --dry-run-prompts-dir artifacts/dry_run_prompts

    # Single-file test against the configured Azure backend
    python scripts/s5/s5_llm_reason.py \
        --config configs/llm_backends.yaml \
        --backend azure-gpt-5-mini \
        --prompt-file scripts/s5/prompts/base_prompt.txt \
        --single-file examples/evidence/synthetic_window.json \
        --output-dir artifacts/s5_single

    # Single-file test against local Ollama backend
    python scripts/s5/s5_llm_reason.py \
        --config configs/llm_backends.yaml \
        --backend ollama-local \
        --prompt-file scripts/s5/prompts/base_prompt.txt \
        --single-file examples/evidence/synthetic_window.json \
        --output-dir artifacts/s5_single_ollama
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        """Fallback iterator used for prompt-only dry runs."""
        return iterable

# We use the new OpenAI SDK for Azure
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None


# ---------------------------- config loading ---------------------------- #

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_environment_variables(value: Any) -> Any:
    """Recursively resolve ``${NAME}`` placeholders without logging values."""
    missing = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.getenv(name)
        if resolved is None:
            missing.add(name)
            return match.group(0)
        return resolved

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            return ENV_PATTERN.sub(replace, item)
        if isinstance(item, list):
            return [visit(entry) for entry in item]
        if isinstance(item, dict):
            return {key: visit(entry) for key, entry in item.items()}
        return item

    expanded = visit(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing environment variables referenced by LLM config: {names}")
    return expanded

def load_llm_backends(config_path: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Load LLM backend configuration YAML.

    Expected structure:
        default_backend: <name>
        backends:
          <name>:
            provider: "azure" | "ollama"
            ... other fields ...
    """
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"LLM config file not found: {config_path}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Invalid LLM config format (expected mapping at top-level).")

    default_backend = cfg.get("default_backend")
    backends = cfg.get("backends", {})

    if not backends:
        raise ValueError("No 'backends' section found in LLM config.")
    if default_backend is None:
        # pick first as default if not specified
        default_backend = next(iter(backends.keys()))

    if default_backend not in backends:
        raise ValueError(f"default_backend '{default_backend}' not found in backends.")

    return default_backend, backends


# ---------------------------- YAML / JSON helpers ---------------------------- #

def llm_input_to_yaml(llm_input: Dict[str, Any]) -> str:
    """
    Convert the llm_input dict to a YAML string.

    We keep keys ordered and allow unicode. No weird escape sequences.
    """
    yaml_str = yaml.safe_dump(
        llm_input,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    yaml_str = yaml_str.encode("utf-8").decode("utf-8")
    return yaml_str


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from LLM response text.

    Strategy:
      1. Look for ```json ... ``` fenced block; take the last one.
      2. If none, try to parse the largest {...} span.
    """
    # 1) Fenced json block
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = None
    if fenced:
        candidate = fenced[-1]
    else:
        # 2) fallback: try from first '{' to last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]

    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------- backend helpers ---------------------------- #

def apply_proxy_if_any(backend_cfg: Dict[str, Any]) -> None:
    """
    If the backend config defines a 'proxy', set HTTP(S)_PROXY env vars.

    This is a pragmatic approach so both openai client & requests use it.
    We do not overwrite existing values if they're already set.
    """
    proxy = backend_cfg.get("proxy")
    if not proxy:
        return

    for key in ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        os.environ.setdefault(key, proxy)


def make_azure_client(backend_cfg: Dict[str, Any]) -> Any:
    """
    Create an AzureOpenAI client from backend config.
    """
    if AzureOpenAI is None:
        raise ImportError(
            "openai package not found. Install with: pip install --upgrade openai"
        )

    endpoint = backend_cfg.get("endpoint")
    api_key = backend_cfg.get("api_key")
    api_version = backend_cfg.get("api_version")
    if not endpoint or not api_key or not api_version:
        raise ValueError(
            "Azure backend requires 'endpoint', 'api_key', and 'api_version' in config."
        )

    apply_proxy_if_any(backend_cfg)

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return client


def call_azure_backend(
    backend_cfg: Dict[str, Any],
    prompt: str,
) -> str:
    """
    Call Azure OpenAI Responses API with a single user prompt.

    We treat the whole formatted prompt as one user message.
    """
    client = make_azure_client(backend_cfg)
    model = backend_cfg.get("model")
    if not model:
        raise ValueError("Azure backend config missing 'model'.")

    params = backend_cfg.get("default_params", {}) or {}
    temperature = float(params.get("temperature", 0.0))
    top_p = float(params.get("top_p", 1.0))
    max_output_tokens = int(params.get("max_output_tokens", 8000))
    # Some SDK versions support seed, some don't; we ignore it to stay robust.
    # seed = params.get("seed", None)

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
    )

    # Extract text from the new Responses API structure.
    # This is robust to multiple content blocks.
    chunks = []
    # New SDKs expose resp.output as a list of messages
    output = getattr(resp, "output", None)
    if output is not None:
        for item in output:
            content_list = getattr(item, "content", []) or []
            for c in content_list:
                # For text blocks, type is often "output_text" with .text.value
                ctype = getattr(c, "type", None)
                if ctype == "output_text":
                    text_obj = getattr(c, "text", None)
                    if text_obj is not None:
                        val = getattr(text_obj, "value", None)
                        if isinstance(val, str):
                            chunks.append(val)
                else:
                    # Fallback: if content has a 'text' attr directly
                    val = getattr(c, "text", None)
                    if isinstance(val, str):
                        chunks.append(val)
    else:
        # Very defensive fallback: try resp.output_text or resp.content if present
        if hasattr(resp, "output_text"):
            chunks.append(str(resp.output_text))
        elif hasattr(resp, "content"):
            chunks.append(str(resp.content))

    full_text = "".join(chunks).strip()
    return full_text


def call_ollama_backend(
    backend_cfg: Dict[str, Any],
    prompt: str,
) -> str:
    """
    Call a local Ollama server via its native /api/chat endpoint.

    Assumes:
        base_url: e.g. "http://127.0.0.1:11434"
        model: e.g. "gpt-oss:20b"
    """
    base_url = backend_cfg.get("base_url", "http://127.0.0.1:11434")
    model = backend_cfg.get("model")
    if not model:
        raise ValueError("Ollama backend config missing 'model'.")

    apply_proxy_if_any(backend_cfg)

    params = backend_cfg.get("default_params", {}) or {}
    temperature = float(params.get("temperature", 0.1))
    top_p = float(params.get("top_p", 0.9))

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    if requests is None:
        raise RuntimeError(
            "The Ollama backend requires the 'llm' dependencies. "
            "Install them with: python -m pip install -e '.[llm]'"
        )

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message", {})
    content = msg.get("content", "")
    return content.strip()


# ---------------------------- core processing ---------------------------- #

PLACEHOLDER = "<<<SCENARIO_YAML>>>"


def process_single_file(
    json_path: Path,
    backend_name: str,
    backend_cfg: Dict[str, Any],
    prompt_template: str,
    out_dir: Path,
    dry_run_prompts_dir: Optional[Path] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Process a single S4.2 JSON file:

      - Load JSON
      - Extract `llm_input`
      - Convert to YAML
      - Replace the placeholder <<<SCENARIO_YAML>>> in prompt_template
      - If dry-run: save prompt and return
      - Else: call selected backend (azure or ollama)
      - Parse JSON from LLM output; ensure ego_window_key present
      - Save *.llm.json file

    Returns:
        (output_json_path or None, error_message or None)
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Failed to read JSON: {e}"

    llm_input = data.get("llm_input")
    if llm_input is None:
        return None, "Missing 'llm_input' in JSON."

    scenario_yaml = llm_input_to_yaml(llm_input)

    if PLACEHOLDER not in prompt_template:
        return None, f"Prompt template must contain the placeholder '{PLACEHOLDER}'."

    # IMPORTANT: use .replace, not .format, to avoid KeyError from {..} in the prompt.
    prompt = prompt_template.replace(PLACEHOLDER, scenario_yaml)

    # Dry-run: only write prompt
    if dry_run_prompts_dir is not None:
        dry_run_prompts_dir.mkdir(parents=True, exist_ok=True)
        out_prompt_path = dry_run_prompts_dir / (json_path.stem + ".prompt.txt")
        with out_prompt_path.open("w", encoding="utf-8") as f:
            f.write(prompt)
        return out_prompt_path, None

    provider = backend_cfg.get("provider")
    if provider not in ("azure", "ollama"):
        return None, f"Unknown provider '{provider}' for backend '{backend_name}'."

    try:
        if provider == "azure":
            raw_text = call_azure_backend(backend_cfg, prompt)
        else:
            raw_text = call_ollama_backend(backend_cfg, prompt)
    except Exception as e:
        return None, f"Backend error: {e}"

    parsed = extract_json_from_text(raw_text)
    if parsed is None:
        # We still save raw text so you can debug prompt/outputs.
        parsed = {
            "error": "Failed to parse JSON from LLM response.",
        }

    # Ensure ego_window_key is present in parsed result (for downstream linking)
    ego_window_key = llm_input.get("ego_window_key")
    if ego_window_key is not None and "ego_window_key" not in parsed:
        parsed["ego_window_key"] = ego_window_key

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (json_path.stem + ".llm.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backend": backend_name,
                "raw_response_text": raw_text,
                "parsed_result": parsed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path, None


# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="S5 – LLM Reasoner for scenario analysis."
    )
    ap.add_argument(
        "--config",
        type=str,
        default="configs/llm_backends.yaml",
        help="Path to LLM backend YAML config.",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Backend name to use (overrides default_backend).",
    )
    ap.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help=f"Path to prompt template file (must contain '{PLACEHOLDER}').",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing S4.2 JSON files.",
    )
    group.add_argument(
        "--single-file",
        type=str,
        help="Single S4.2 JSON file to process.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store LLM output JSON files.",
    )
    ap.add_argument(
        "--dry-run-prompts-dir",
        type=str,
        default=None,
        help="If set, write prompts only (no LLM calls) into this directory.",
    )

    args = ap.parse_args()

    # Load backends
    default_backend_name, backends = load_llm_backends(args.config)
    backend_name = args.backend or default_backend_name

    if backend_name not in backends:
        raise ValueError(
            f"Backend '{backend_name}' not found in config. "
            f"Available: {list(backends.keys())}"
        )

    # Resolve secrets only for the selected backend. This keeps the local
    # Ollama backend usable when Azure variables are intentionally unset.
    backend_cfg = expand_environment_variables(backends[backend_name])

    # Load prompt template
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    prompt_template = prompt_path.read_text(encoding="utf-8")

    out_dir = Path(args.output_dir)
    dry_run_prompts_dir = Path(args.dry_run_prompts_dir) if args.dry_run_prompts_dir else None

    # Determine input files
    if args.single_file:
        json_paths = [Path(args.single_file)]
    else:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {in_dir}")
        json_paths = sorted(in_dir.glob("*.json"))

    if not json_paths:
        print("No JSON files found to process.")
        return

    print(f"[S5] Using backend: {backend_name} (provider={backend_cfg.get('provider')})")
    if dry_run_prompts_dir is not None:
        print(f"[S5] DRY-RUN mode: writing prompts to {dry_run_prompts_dir} (no LLM calls)")
    print(f"[S5] Input files: {len(json_paths)}")
    print(f"[S5] Output dir: {out_dir}")

    n_ok = 0
    n_err = 0

    for jp in tqdm(json_paths, desc="Processing scenarios"):
        out_path, err = process_single_file(
            jp,
            backend_name=backend_name,
            backend_cfg=backend_cfg,
            prompt_template=prompt_template,
            out_dir=out_dir,
            dry_run_prompts_dir=dry_run_prompts_dir,
        )
        if err:
            n_err += 1
            print(f"[ERROR] {jp.name}: {err}")
        else:
            n_ok += 1

    print(f"\n[S5] Done. OK={n_ok}, errors={n_err}")


if __name__ == "__main__":
    main()
