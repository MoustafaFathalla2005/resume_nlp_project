import os
import re

# ── default path ─────────────────────────────────────────────────────────────

_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "jd_generator_db.txt")

# ── low-level helpers ─────────────────────────────────────────────────────────

def _read_section(filepath: str, section_name: str) -> str:

    text = open(filepath, encoding="utf-8").read()
    pattern = rf"###BEGIN_{re.escape(section_name)}###\n(.*?)###END_{re.escape(section_name)}###"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Section '{section_name}' not found in '{filepath}'.\n"
            f"Expected delimiters:  ###BEGIN_{section_name}###  /  ###END_{section_name}###"
        )
    return match.group(1)


def _parse_key_blocks(section_text: str) -> dict:
    result = {}
    pattern = r"---KEY:\s*(.+?)---\n(.*?)---ENDKEY---"
    for block in re.finditer(pattern, section_text, re.DOTALL):
        key    = block.group(1).strip()
        lines  = [ln.strip() for ln in block.group(2).splitlines() if ln.strip()]
        result[key] = lines
    return result

# ── public readers ────────────────────────────────────────────────────────────

def get_skills_db(filepath: str = _DEFAULT_DB) -> dict[str, list[str]]:

    section = _read_section(filepath, "SKILLS_DB")
    return _parse_key_blocks(section)


def get_category_title_map(filepath: str = _DEFAULT_DB) -> dict[str, str]:

    section = _read_section(filepath, "CATEGORY_TITLE_MAP")
    raw     = _parse_key_blocks(section)
    # each block has exactly one title line — unwrap the list
    return {cat: lines[0] for cat, lines in raw.items() if lines}


def get_summaries(filepath: str = _DEFAULT_DB) -> dict[str, str]:

    section = _read_section(filepath, "SUMMARIES")
    raw     = _parse_key_blocks(section)
    # join lines back into one string (template text may span lines in the file)
    return {cat: " ".join(lines) for cat, lines in raw.items() if lines}


def get_responsibilities(filepath: str = _DEFAULT_DB) -> dict[str, list[str]]:

    section = _read_section(filepath, "RESPONSIBILITIES")
    return _parse_key_blocks(section)


def get_offers(filepath: str = _DEFAULT_DB) -> dict[str, list[str]]:

    section = _read_section(filepath, "OFFERS")
    return _parse_key_blocks(section)
