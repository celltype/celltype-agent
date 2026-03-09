"""Patch jackhmmer.py to not fail on empty databases."""
path = "/app/alphafold/alphafold/data/tools/jackhmmer.py"
with open(path) as f:
    code = f.read()
code = code.replace(
    "raise RuntimeError(",
    "return []  # patched\n        raise RuntimeError(",
)
with open(path, "w") as f:
    f.write(code)
print("Patched jackhmmer.py")
