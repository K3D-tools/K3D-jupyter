"""Guard the `requires-python = ">=3.7"` claim in pyproject.toml.

The package advertises Python 3.7 support but CI runs newer interpreters, so nothing otherwise
stops a 3.8+-only construct from slipping in and breaking `import k3d` on 3.7 (this happened
once already with `typing.Literal`). These checks need no 3.7 interpreter: they parse the
sources with `feature_version=(3, 7)` and inspect imports.

Version-guarded imports are fine and must not be flagged - that is exactly how
`k3d/_protocol.py` keeps `typing.Literal` while still importing on 3.7. So the import check is
AST-based rather than a text search: an import only counts if it executes unconditionally,
i.e. it is not inside an `if sys.version_info ...` branch or a `try/except ImportError`.

If the project ever drops 3.7, delete this file and raise requires-python instead.
"""

import ast
import os
import re

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Names importable from these modules only on 3.8+.
POST_37_FROM = {
    "typing": {"Literal", "Protocol", "TypedDict", "Final", "runtime_checkable"},
}

# Whole modules that do not exist in the 3.7 stdlib.
POST_37_MODULES = {
    "zoneinfo": "3.9+",
    "graphlib": "3.9+",
    "importlib.metadata": "3.8+",
}

# Attribute / method usages that no import statement reveals.
POST_37_USAGES = {
    r"\bfunctools\.cached_property\b": "functools.cached_property (3.8+)",
    r"\bmath\.prod\b": "math.prod (3.8+)",
    r"\.removeprefix\(": "str.removeprefix (3.9+)",
    r"\.removesuffix\(": "str.removesuffix (3.9+)",
}


def _sources():
    for dirpath, dirnames, filenames in os.walk(PACKAGE_ROOT):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"test", "static", "labextension", "__pycache__"}
        ]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


SOURCES = sorted(_sources())


def _is_version_guard(test):
    """True for `sys.version_info ...` and `typing.TYPE_CHECKING` style conditions."""
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute) and node.attr in (
                "version_info",
                "TYPE_CHECKING",
        ):
            return True
        if isinstance(node, ast.Name) and node.id in ("version_info", "TYPE_CHECKING"):
            return True
    return False


def _catches_import_error(handlers):
    for handler in handlers:
        if handler.type is None:  # bare except
            return True
        for node in ast.walk(handler.type):
            if isinstance(node, ast.Name) and node.id in (
                    "ImportError",
                    "ModuleNotFoundError",
                    "Exception",
            ):
                return True
    return False


def _import_label(node):
    """Label if this import needs Python > 3.7, else None."""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module in POST_37_MODULES:
            return "%s (%s)" % (module, POST_37_MODULES[module])
        hits = sorted({alias.name for alias in node.names} & POST_37_FROM.get(module, set()))
        if hits:
            return "%s.%s (3.8+)" % (module, ", ".join(hits))
    else:
        for alias in node.names:
            if alias.name in POST_37_MODULES:
                return "%s (%s)" % (alias.name, POST_37_MODULES[alias.name])
    return None


def _unguarded_post_37_imports(tree):
    problems = []

    def walk(statements, guarded):
        for node in statements:
            if isinstance(node, ast.If):
                inner = guarded or _is_version_guard(node.test)
                walk(node.body, inner)
                walk(node.orelse, inner)
                continue

            if isinstance(node, ast.Try):
                inner = guarded or _catches_import_error(node.handlers)
                walk(node.body, inner)
                for handler in node.handlers:
                    walk(handler.body, inner)
                walk(node.orelse, inner)
                walk(node.finalbody, guarded)
                continue

            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not guarded:
                    label = _import_label(node)
                    if label:
                        problems.append("line %d: %s" % (node.lineno, label))
                continue

            for field in ("body", "orelse", "finalbody"):
                body = getattr(node, field, None)
                if isinstance(body, list):
                    walk([s for s in body if isinstance(s, ast.stmt)], guarded)

    walk(tree.body, False)
    return problems


def test_sources_were_found():
    # A silently empty file list would make every check below pass vacuously.
    assert len(SOURCES) > 10, SOURCES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: os.path.basename(p))
def test_parses_as_python_37_syntax(path):
    with open(path, encoding="utf-8") as f:
        source = f.read()

    try:
        ast.parse(source, filename=path, feature_version=(3, 7))
    except SyntaxError as e:
        pytest.fail(
            "%s uses syntax newer than Python 3.7 (line %s): %s"
            % (os.path.relpath(path, PACKAGE_ROOT), e.lineno, e.msg)
        )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: os.path.basename(p))
def test_no_unguarded_post_37_imports(path):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)

    problems = _unguarded_post_37_imports(tree)

    assert not problems, (
        "%s imports names missing from the Python 3.7 stdlib without a version guard: %s. "
        "Wrap them in `if sys.version_info >= (3, 8):` with a 3.7 fallback."
        % (os.path.relpath(path, PACKAGE_ROOT), "; ".join(problems))
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: os.path.basename(p))
def test_no_post_37_attribute_usage(path):
    with open(path, encoding="utf-8") as f:
        source = f.read()

    problems = []
    for pattern, label in POST_37_USAGES.items():
        for match in re.finditer(pattern, source):
            line = source[: match.start()].count("\n") + 1
            problems.append("line %d: %s" % (line, label))

    assert not problems, "%s uses attributes missing in Python 3.7: %s" % (
        os.path.relpath(path, PACKAGE_ROOT),
        "; ".join(problems),
    )
