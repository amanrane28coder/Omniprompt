"""Module 4: immutable versioning, branching, rollback-by-copy-forward.

Versions are never edited. Branching and rollback both create a NEW version
whose parent_version_id points at the source version; rollback copies the
source template text, branching starts from it as well.
"""


def next_version_no(existing_version_numbers: list[int]) -> int:
    return (max(existing_version_numbers) + 1) if existing_version_numbers else 1


def branch_template(source_template_text: str) -> str:
    return source_template_text


def rollback_template(source_template_text: str) -> str:
    # Copy-forward: the old version stays untouched; the new version is
    # identical in text but has a new id, version_no and parent pointer.
    return source_template_text
