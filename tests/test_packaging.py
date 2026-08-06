"""Every module the app imports has to reach the container.

The Dockerfile copies source files one by one rather than the whole directory, so
adding a module and forgetting its COPY line builds an image that only fails once
someone runs it.
"""

import ast
import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _local_modules():
    return {
        name[:-3] for name in os.listdir(APP_DIR)
        if name.endswith(".py") and not name.startswith("_")
    }


def _imported_by(module, local):
    """The local modules ``module`` imports, one level deep."""
    tree = ast.parse((open(os.path.join(APP_DIR, f"{module}.py")).read()))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    return imported & local


def _reachable_from_entry_point():
    """Every local module reachable from main.py, transitively."""
    local = _local_modules()
    seen, pending = set(), ["main"]
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        pending.extend(_imported_by(module, local) - seen)
    return seen


def test_dockerfile_copies_every_module_the_app_imports():
    dockerfile = open(os.path.join(APP_DIR, "Dockerfile")).read()

    missing = sorted(
        module for module in _reachable_from_entry_point()
        if f"COPY {module}.py" not in dockerfile
    )

    assert not missing, f"Dockerfile has no COPY line for: {', '.join(missing)}"


def test_dockerfile_does_not_copy_modules_that_no_longer_exist():
    dockerfile = open(os.path.join(APP_DIR, "Dockerfile")).read()

    stale = [
        line.split()[1] for line in dockerfile.splitlines()
        if line.startswith("COPY ") and line.split()[1].endswith(".py")
        and not os.path.exists(os.path.join(APP_DIR, line.split()[1]))
    ]

    assert not stale, f"Dockerfile copies files that are gone: {', '.join(stale)}"
