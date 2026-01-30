"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution, SourceFile


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _resolve_entry_point(source_dir: Path, entry_point: str, language: str) -> str:
    """Resolve entry point to '<relative_file>::<function>' format."""
    if "::" in entry_point:
        return entry_point

    if language in {"triton", "python"}:
        exts = {".py"}
    elif language == "cuda":
        exts = {".cu", ".cpp", ".cc", ".cxx", ".c"}
    else:
        raise ValueError(f"Unsupported language: {language}")

    candidates = sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix in exts
        ]
    )
    if not candidates:
        raise FileNotFoundError(f"No source files found under: {source_dir}")

    preferred = [p for p in candidates if p.stem == "kernel"]
    if preferred:
        entry_file = preferred[0]
    elif len(candidates) == 1:
        entry_file = candidates[0]
    else:
        raise ValueError(
            "Multiple source files found. Set build.entry_point to '<file>::<fn>'."
        )

    rel_path = entry_file.relative_to(source_dir).as_posix()
    return f"{rel_path}::{entry_point}"


def _collect_sources(source_dir: Path) -> list[SourceFile]:
    sources: list[SourceFile] = []
    skip_dirs = {"__pycache__", ".git", ".pytest_cache"}
    skip_exts = {".pyc", ".pyo", ".so", ".o"}
    for path in sorted([p for p in source_dir.rglob("*") if p.is_file()]):
        # Skip cache directories and binary files
        if any(d in path.parts for d in skip_dirs):
            continue
        if path.suffix in skip_exts:
            continue
        rel_path = path.relative_to(source_dir).as_posix()
        sources.append(SourceFile(path=rel_path, content=path.read_text()))
    return sources


def pack_solution(output_path: Path = None) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_point = build_config["entry_point"]

    # Determine source directory based on language
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    elif language == "python":
        source_dir = PROJECT_ROOT / "solution" / "python"
    else:
        raise ValueError(f"Unsupported language: {language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    resolved_entry_point = _resolve_entry_point(source_dir, entry_point, language)
    sources = _collect_sources(source_dir)

    # Create build spec
    spec = BuildSpec(
        language=language,
        target_hardware=["cuda"],
        entry_point=resolved_entry_point,
    )

    solution = Solution(
        name=solution_config["name"],
        definition=solution_config["definition"],
        author=solution_config["author"],
        spec=spec,
        sources=sources,
        description=solution_config.get(
            "description", f"Custom kernel for {solution_config['definition']}"
        ),
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {language}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    args = parser.parse_args()

    try:
        pack_solution(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
