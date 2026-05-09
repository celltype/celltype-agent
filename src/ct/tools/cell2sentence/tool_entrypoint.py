#!/usr/bin/env python3
"""Standard container entrypoint for all GPU/CPU tools."""

import json
import os
import sys
import traceback


def main():
    input_file = os.environ.get("INPUT_FILE", "/workspace/input.json")
    output_file = os.environ.get("OUTPUT_FILE", "/workspace/output.json")
    try:
        with open(input_file) as f:
            args = json.load(f)
        if not isinstance(args, dict):
            args = {}
        sys.path.insert(0, "/opt")
        from implementation import run

        result = run(**args)
        if not isinstance(result, dict):
            result = {"summary": str(result)}
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as exc:
        with open(output_file, "w") as f:
            json.dump({"summary": f"Error: {exc}", "error": traceback.format_exc()}, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
