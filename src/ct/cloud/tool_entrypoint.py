#!/usr/bin/env python
"""
Standard container entrypoint for GPU/high-memory tools.

Protocol:
  1. Read tool arguments from INPUT_FILE (default: /workspace/input.json)
  2. Import and call implementation.run(**args)
  3. Write result dict to OUTPUT_FILE (default: /workspace/output.json)

On error, writes {"summary": "Error: ...", "error": "<traceback>"} and exits with code 1.
"""

import json
import os
import sys
import traceback


def main():
    input_file = os.environ.get("INPUT_FILE", "/workspace/input.json")
    output_file = os.environ.get("OUTPUT_FILE", "/workspace/output.json")

    try:
        # Read input arguments
        with open(input_file) as f:
            args = json.load(f)

        if not isinstance(args, dict):
            args = {}

        # Import and run implementation
        sys.path.insert(0, "/opt")
        from implementation import run

        result = run(**args)

        if not isinstance(result, dict):
            result = {"summary": str(result)}

        # Write output
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

    except Exception as e:
        tb = traceback.format_exc()
        error_result = {
            "summary": f"Error: {e}",
            "error": tb,
        }
        with open(output_file, "w") as f:
            json.dump(error_result, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
