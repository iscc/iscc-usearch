"""Generate llms-full.txt and per-page .md files for LLM consumption."""

import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"
SITE_DIR = Path(__file__).parent.parent / "site"

# Ordered list of doc pages to include (relative to docs/)
PAGES = [
    "index.md",
    "tutorials/getting-started.md",
    "tutorials/variable-length.md",
    "tutorials/scaling-up.md",
    "howto/persistence.md",
    "howto/sharding.md",
    "howto/uuid-keys.md",
    "howto/upsert.md",
    "howto/bloom-filters.md",
    "explanation/nphd-metric.md",
    "explanation/architecture.md",
    "explanation/sharding-design.md",
    "explanation/performance.md",
    "reference/api.md",
    "reference/for-coding-agents.md",
    "development/contributing.md",
]

# Page requiring dynamic generation from Python source
API_PAGE = "reference/api.md"

# Regex to strip YAML frontmatter
FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Regex to strip snippet auto-append directives
SNIPPET_RE = re.compile(r"^\*\[.*?\]:.*$", re.MULTILINE)


def strip_frontmatter(content):
    """Remove YAML frontmatter from markdown content."""
    return FRONTMATTER_RE.sub("", content)


def strip_snippets(content):
    """Remove abbreviation snippet definitions appended by pymdownx.snippets."""
    return SNIPPET_RE.sub("", content)


def clean_content(content):
    """Strip frontmatter, snippets, and normalize whitespace."""
    content = strip_frontmatter(content)
    content = strip_snippets(content)
    return content.strip()


# --- API Reference Generation from Python Source ---

# Classes to document with brief descriptions (matching docs/reference/api.md)
_API_CLASSES = [
    ("NphdIndex", "Single-file index for variable-length binary bit-vectors with NPHD metric."),
    (
        "ShardedNphdIndex",
        "Multi-shard index combining automatic sharding with NPHD support for variable-length vectors.",
    ),
    ("ShardedIndex", "Generic sharded index for any metric. Use `ShardedNphdIndex` for NPHD workloads."),
    (
        "ShardedIndex128",
        "Sharded index with 128-bit UUID keys. Uses `bytes(16)` for single keys and `np.dtype('V16')` arrays for batches.",
    ),
    ("ShardedNphdIndex128", "Sharded NPHD index with 128-bit UUID keys for variable-length vectors."),
    ("ScalableBloomFilter", "Scalable bloom filter for efficient probabilistic key existence checks."),
    ("timer", "Context manager for timing operations with loguru integration."),
]


def generate_api_reference():
    """Generate API reference markdown from Python source using griffe."""
    import griffe

    src_path = str(DOCS_DIR.parent / "src")
    package = griffe.load("iscc_usearch", search_paths=[src_path])

    lines = [
        "# API Reference",
        "",
        "Auto-generated documentation for all public classes in `iscc-usearch`.",
    ]

    for class_name, brief in _API_CLASSES:
        obj = package.members.get(class_name)
        if obj is None:
            print(f"Warning: {class_name} not found in package, skipping")
            continue
        lines.append("")
        lines.extend(_render_class(obj, brief))

    return "\n".join(lines)


def _render_class(cls_obj, brief):
    """Render a class as markdown with constructor, properties, and methods."""
    lines = [f"## {cls_obj.name}", "", brief, ""]

    # Class docstring
    cls_sections = _parse_docstring(cls_obj.docstring) if cls_obj.docstring else {}
    if cls_sections.get("text"):
        lines.append(cls_sections["text"])
        lines.append("")

    # Constructor signature and parameter docs
    init = cls_obj.members.get("__init__") if hasattr(cls_obj, "members") else None
    if init and hasattr(init, "parameters"):
        sig = _format_signature(cls_obj.name, init)
        lines.extend(["```python", sig, "```", ""])

        # Parameter docs: prefer init docstring, fall back to class docstring
        init_sections = _parse_docstring(init.docstring) if init.docstring else {}
        params = init_sections.get("params", [])
        if not params:
            params = cls_sections.get("params", [])

        if params:
            for p in params:
                lines.append(f"- **{p['name']}**: {p['desc']}")
            lines.append("")

    # Public members (properties and methods) in source order
    if hasattr(cls_obj, "members"):
        for mname, member in cls_obj.members.items():
            if mname.startswith("_"):
                continue
            if hasattr(member, "parameters"):
                lines.extend(_render_method(member))
                lines.append("")
            elif member.docstring:
                # Only render attributes/properties that have docstrings
                lines.extend(_render_property(member))
                lines.append("")

    return lines


def _format_signature(name, func):
    """Format a callable signature as 'name(param=default, ...)'."""
    parts = []
    saw_kw_marker = False

    # Check if any var_positional (*args) parameter exists
    has_var_pos = any(p.kind.name == "var_positional" for p in func.parameters if p.name not in ("self", "cls"))

    for param in func.parameters:
        if param.name in ("self", "cls"):
            continue

        kind = param.kind.name

        # Insert bare * before first keyword-only param when no *args exists
        if kind == "keyword_only" and not saw_kw_marker and not has_var_pos:
            parts.append("*")
            saw_kw_marker = True

        if kind == "var_positional":
            part = f"*{param.name}"
        elif kind == "var_keyword":
            part = f"**{param.name}"
        else:
            part = param.name
            if param.default is not None:
                default_str = str(param.default)
                # Suppress unhelpful defaults: empty dict for **kwargs, None for required params
                if default_str and default_str not in ("{}",):
                    part += f"={default_str}"

        parts.append(part)

    return f"{name}({', '.join(parts)})"


def _render_property(prop):
    """Render a property or documented attribute as markdown."""
    lines = [f"### *property* {prop.name}"]

    if prop.docstring:
        sections = _parse_docstring(prop.docstring)
        if sections.get("text"):
            lines.extend(["", sections["text"]])
        if sections.get("returns"):
            lines.extend(["", f"**Returns:** {sections['returns']}"])

    return lines


def _render_method(method):
    """Render a method as markdown with signature and docs."""
    # Detect static/class method via labels
    prefix = ""
    labels = getattr(method, "labels", set())
    if "staticmethod" in labels:
        prefix = "*staticmethod* "
    elif "classmethod" in labels:
        prefix = "*classmethod* "

    sig = _format_signature(method.name, method)
    lines = [f"### {prefix}{method.name}", "", "```python", sig, "```"]

    if method.docstring:
        sections = _parse_docstring(method.docstring)
        if sections.get("text"):
            lines.extend(["", sections["text"]])
        if sections.get("params"):
            lines.append("")
            for p in sections["params"]:
                lines.append(f"- **{p['name']}**: {p['desc']}")
        if sections.get("returns"):
            lines.extend(["", f"**Returns:** {sections['returns']}"])
        if sections.get("raises"):
            lines.append("")
            for r in sections["raises"]:
                lines.append(f"**Raises:** `{r['exc']}` — {r['desc']}")

    return lines


def _parse_docstring(docstring):
    """Parse a Sphinx-style docstring into structured sections."""
    result = {"text": "", "params": [], "returns": "", "raises": []}
    if not docstring:
        return result

    try:
        parsed = docstring.parse("sphinx")
    except Exception:
        result["text"] = docstring.value.strip()
        return result

    for section in parsed:
        kind = section.kind.value
        if kind == "text":
            if result["text"]:
                result["text"] += "\n\n" + section.value.strip()
            else:
                result["text"] = section.value.strip()
        elif kind == "parameters":
            for param in section.value:
                desc = param.description.strip() if param.description else ""
                result["params"].append({"name": param.name, "desc": desc})
        elif kind == "returns":
            if section.value:
                descs = [r.description.strip() for r in section.value if r.description]
                result["returns"] = " ".join(descs)
        elif kind == "raises":
            for exc in section.value:
                annotation = str(exc.annotation) if exc.annotation else "Exception"
                desc = exc.description.strip() if exc.description else ""
                result["raises"].append({"exc": annotation, "desc": desc})

    return result


def main():
    """Generate llms-full.txt and individual .md files from doc sources."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    parts = []

    for page in PAGES:
        if page == API_PAGE:
            content = generate_api_reference()
        else:
            path = DOCS_DIR / page
            if not path.exists():
                print(f"Warning: {page} not found, skipping")
                continue
            content = clean_content(path.read_text(encoding="utf-8"))

        if not content:
            continue
        parts.append(content)

        # Write individual .md file to site directory
        md_path = SITE_DIR / page
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(content + "\n", encoding="utf-8")

    # Write concatenated llms-full.txt
    output = "\n\n---\n\n".join(parts) + "\n"
    out_path = SITE_DIR / "llms-full.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"Generated {out_path} ({len(parts)} pages, {len(output)} bytes)")
    print(f"Generated {len(parts)} individual .md files in {SITE_DIR}")


if __name__ == "__main__":
    main()
