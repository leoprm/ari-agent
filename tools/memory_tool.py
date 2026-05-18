#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove, read
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"

ENTRY_DELIMITER = "\n§\n"

# ---------------------------------------------------------------------------
# Hash ID helpers — deterministic identity for memory entries
# ---------------------------------------------------------------------------

_HASH_PREFIX_RE = re.compile(r'^\[h:([0-9a-f]{8})\]\s')


def _hash_content(content: str) -> str:
    """SHA256 first 8 chars — deterministic ID for content."""
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def _prefix_entry(content: str) -> str:
    """Prepend [h:XXXXXXXX] to entry content."""
    h = _hash_content(content)
    return f"[h:{h}] {content}"


def _strip_hash_prefix(entry: str) -> str:
    """Remove [h:XXXXXXXX] prefix if present, return raw content."""
    m = _HASH_PREFIX_RE.match(entry)
    if m:
        return entry[m.end():]
    return entry


# ---------------------------------------------------------------------------
# JSONL sidecar — DAG de memoria con historial inmutable
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL entries. Returns [] if file doesn't exist."""
    if not path.exists():
        return []
    try:
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
    except (OSError, IOError, json.JSONDecodeError):
        return []


def _append_jsonl(path: Path, new_entries: list[dict]):
    """Append entries to JSONL file. Creates file if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False) for e in new_entries]
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
# ---------------------------------------------------------------------------

_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration via curl/wget with secrets
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence via shell rc
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env"),
]

# Subset of invisible chars for injection detection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X} (possible injection)."

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pid}'. Memory entries are injected into the system prompt and must not contain injection or exfiltration payloads."

    return None


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375,
                 lessons_char_limit: int = 3000, memory_dir: Optional[str] = None):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.lessons_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.lessons_char_limit = lessons_char_limit
        # tree_memory_path: when set, enables dual global(read-only) + tree(read-write) mode
        self.tree_memory_path = Path(memory_dir) if memory_dir else None
        # Frozen snapshots for system prompt -- set once at load_from_disk()
        # When tree_memory_path is set:
        #   _system_prompt_snapshot holds the TREE (read-write) blocks
        #   _global_snapshot holds the GLOBAL (read-only) blocks
        # When tree_memory_path is None:
        #   _system_prompt_snapshot holds the global blocks (standard mode)
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": "", "lessons": ""}
        self._global_snapshot: Dict[str, str] = {"memory": "", "user": "", "lessons": ""}

    def load_from_disk(self):
        """Load entries from JSONL (primary) or MD (fallback), capture system prompt snapshot.

        When tree_memory_path is set (dual mode):
          1. Load global memory from get_memory_dir() — frozen, read-only snapshot
          2. Load tree memory from self.tree_memory_path — read-write live state
        When tree_memory_path is None (standard mode):
          Load global memory as the only state (unchanged behavior).
        """
        global_mem_dir = get_memory_dir()
        global_mem_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: Load global memory (always, as read-only reference) ──
        global_entries: Dict[str, List[str]] = {}
        for target in ("memory", "user", "lessons"):
            jsonl_path = self._jsonl_path_for(target, use_tree=False)
            md_path = self._path_for(target, use_tree=False)

            if jsonl_path.exists():
                # Read entries from JSONL for this target (read-only snapshot)
                raw_entries = _read_jsonl(jsonl_path)
                # Reconstruct active entries from JSONL DAG
                entries = self._active_entries_from_jsonl(raw_entries) if raw_entries else []
            elif md_path.exists():
                entries = self._read_file(md_path)
                migrated = self._migrate_legacy_entries(entries)
                if migrated:
                    entries = migrated
            else:
                entries = []

            entries = list(dict.fromkeys(entries))  # deduplicate
            global_entries[target] = entries

        # Always capture global snapshot (used in both modes)
        self._global_snapshot = {
            "memory": self._render_block("memory", global_entries["memory"]),
            "user": self._render_block("user", global_entries["user"]),
            "lessons": self._render_block("lessons", global_entries["lessons"]),
        }

        # ── Phase 2: Load tree memory (read-write) ──
        if self.tree_memory_path:
            self.tree_memory_path.mkdir(parents=True, exist_ok=True)
            for target in ("memory", "user", "lessons"):
                jsonl_path = self._jsonl_path_for(target, use_tree=True)
                md_path = self._path_for(target, use_tree=True)

                if jsonl_path.exists():
                    self._rebuild_md_from_jsonl(target, use_tree=True)
                elif md_path.exists():
                    entries = self._read_file(md_path)
                    migrated = self._migrate_legacy_entries(entries)
                    if migrated:
                        entries = migrated
                    self._set_entries(target, entries)
                    self.save_to_disk(target)
                else:
                    self._set_entries(target, [])

            # If tree hasn't been written yet (no JSONL), seed from MD file reads
            for target in ("memory", "user", "lessons"):
                entries = self._entries_for(target)
                entries = list(dict.fromkeys(entries))
                self._set_entries(target, entries)
        else:
            # Standard mode: global entries ARE the live entries
            for target in ("memory", "user", "lessons"):
                entries = global_entries[target]
                entries = list(dict.fromkeys(entries))
                self._set_entries(target, entries)
                # Trigger JSONL migration if MD-only entries exist
                jsonl_path = self._jsonl_path_for(target, use_tree=False)
                md_path = self._path_for(target, use_tree=False)
                if not jsonl_path.exists() and md_path.exists():
                    self.save_to_disk(target)

        # Capture tree snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
            "lessons": self._render_block("lessons", self.lessons_entries),
        }

    @staticmethod
    def _migrate_legacy_entries(entries: List[str]) -> Optional[List[str]]:
        """Detect entries without [h:...] prefix and assign hashes.

        Returns a new list with prefixed entries if any were migrated,
        or None if all entries already have hash prefixes.
        """
        migrated = False
        result = []
        for entry in entries:
            if _HASH_PREFIX_RE.match(entry):
                result.append(entry)
            else:
                raw = _strip_hash_prefix(entry)
                result.append(_prefix_entry(raw))
                migrated = True
        return result if migrated else None

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    def _path_for(self, target: str, use_tree: bool = False) -> Path:
        """Return the MD file path for a target.

        When use_tree=True and self.tree_memory_path is set, returns paths
        under tree_memory_path. For trees, 'lessons' target maps to
        memory_private.md.
        """
        if use_tree and self.tree_memory_path:
            if target == "user":
                return self.tree_memory_path / "USER.md"
            if target == "lessons":
                return self.tree_memory_path / "memory_private.md"
            return self.tree_memory_path / "MEMORY.md"

        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        if target == "lessons":
            return mem_dir / "LECCIONES.md"
        return mem_dir / "MEMORY.md"

    def _jsonl_path_for(self, target: str, use_tree: bool = False) -> Path:
        """Return the JSONL sidecar path for a target.

        Routes to tree_memory_path when use_tree=True and tree mode is active.
        """
        if use_tree and self.tree_memory_path:
            if target == "user":
                return self.tree_memory_path / "USER.jsonl"
            if target == "lessons":
                return self.tree_memory_path / "memory_private.jsonl"
            return self.tree_memory_path / "MEMORY.jsonl"

        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.jsonl"
        if target == "lessons":
            return mem_dir / "LECCIONES.jsonl"
        return mem_dir / "MEMORY.jsonl"

    def _active_entries_from_jsonl(self, entries_data: list) -> List[str]:
        """Given raw JSONL entries, return active (non-superseded) entries as [h:...] strings."""
        superseded: set[str] = set()
        retired: set[str] = set()
        for entry in entries_data:
            pid = entry.get("parent_id")
            if pid:
                superseded.add(pid)
            if entry.get("action") in ("remove", "merged"):
                retired.add(entry["id"])

        active = []
        seen_ids: set[str] = set()
        for entry in entries_data:
            if entry["id"] in retired:
                continue
            if entry.get("action") in ("add", "replace", "migrate", "merge") and entry["id"] not in superseded:
                if entry["id"] not in seen_ids:
                    active.append(f"[h:{entry['id']}] {entry.get('content', '')}")
                    seen_ids.add(entry["id"])
        return active

    def _rebuild_md_from_jsonl(self, target: str, use_tree: bool = False):
        """Regenerate MEMORY.md / USER.md from JSONL (active entries only).

        Active = entries whose id has NOT been superseded by a later
        replace or remove (i.e. id does not appear as any parent_id).
        """
        jsonl_path = self._jsonl_path_for(target, use_tree=use_tree)
        if not jsonl_path.exists():
            return

        entries_data = _read_jsonl(jsonl_path)
        if not entries_data:
            return

        md_entries = self._active_entries_from_jsonl(entries_data)

        md_path = self._path_for(target, use_tree=use_tree)
        self._write_file(md_path, md_entries)

        # Keep in-memory state in sync (only for tree mode — global is ref only)
        if use_tree or not self.tree_memory_path:
            self._set_entries(target, md_entries)

    def _reload_target(self, target: str):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        When JSONL sidecar exists, source of truth is JSONL.
        Routes to tree paths when tree_memory_path is set.
        """
        use_tree = bool(self.tree_memory_path)
        jsonl_path = self._jsonl_path_for(target, use_tree=use_tree)
        if jsonl_path.exists():
            self._rebuild_md_from_jsonl(target, use_tree=use_tree)
        else:
            fresh = self._read_file(self._path_for(target, use_tree=use_tree))
            fresh = list(dict.fromkeys(fresh))  # deduplicate
            self._set_entries(target, fresh)

    def save_to_disk(self, target: str, action: str = None, entry_id: str = None,
                     content: str = None, parent_id: str = None):
        """Persist entries: append JSONL entry (if mutation), regenerate MD.

        With JSONL sidecar:
          - Append one line for this mutation
          - Rebuild MD from JSONL (active entries only)
        Without JSONL sidecar (legacy):
          - Write MD directly
          - Create initial JSONL from current entries (orphan:true)

        Routes to tree paths when tree_memory_path is set.
        """
        use_tree = bool(self.tree_memory_path)
        target_dir = self.tree_memory_path if use_tree else get_memory_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._jsonl_path_for(target, use_tree=use_tree)

        if action and entry_id:
            # Normal mutation: append to JSONL
            jsonl_entry = {
                "id": entry_id,
                "content": content or "",
                "parent_id": parent_id,
                "action": action,
                "ts": int(time.time()),
                "session_id": os.environ.get("HERMES_SESSION_ID", ""),
                "orphan": False,
            }
            _append_jsonl(jsonl_path, [jsonl_entry])
            self._rebuild_md_from_jsonl(target, use_tree=use_tree)
        elif not jsonl_path.exists():
            # First time with legacy .md: migrate all current entries to JSONL
            entries = self._entries_for(target)
            if entries:
                jsonl_entries = []
                for entry_text in entries:
                    raw = _strip_hash_prefix(entry_text)
                    h = _hash_content(raw)
                    jsonl_entries.append({
                        "id": h,
                        "content": raw,
                        "parent_id": None,
                        "action": "migrate",
                        "ts": int(time.time()),
                        "session_id": os.environ.get("HERMES_SESSION_ID", ""),
                        "orphan": True,
                    })
                _append_jsonl(jsonl_path, jsonl_entries)
            # Write MD files as well (already have the entries in memory)
            self._write_file(self._path_for(target, use_tree=use_tree), entries)
        else:
            # JSONL exists, no specific mutation — just sync MD
            self._rebuild_md_from_jsonl(target, use_tree=use_tree)

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        if target == "lessons":
            return self.lessons_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        elif target == "lessons":
            self.lessons_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        if target == "lessons":
            return self.lessons_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        prefixed = _prefix_entry(content)
        entry_id = _hash_content(content)
        use_tree = bool(self.tree_memory_path)

        with self._file_lock(self._path_for(target, use_tree=use_tree)):
            # Re-read from disk under lock to pick up writes from other sessions
            self._reload_target(target)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if prefixed in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).", entry_id=entry_id)

            # Calculate what the new total would be
            new_entries = entries + [prefixed]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return {
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Replace or remove existing entries first."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                }

            entries.append(prefixed)
            self._set_entries(target, entries)
            self.save_to_disk(target, action="add", entry_id=entry_id,
                              content=content)

        return self._success_response(target, "Entry added.", entry_id=entry_id)

    def replace(self, target: str, old_text: str, new_content: str,
                parent: str = None) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content.

        If parent is provided, it's the hash ID of the entry being replaced.
        If omitted, auto-detects via substring match in current entries.
        """
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        prefixed_new = _prefix_entry(new_content)
        new_hash = _hash_content(new_content)
        use_tree = bool(self.tree_memory_path)

        with self._file_lock(self._path_for(target, use_tree=use_tree)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            # Use explicit parent hash if provided, otherwise auto-detect from content
            if parent:
                old_hash = parent
            else:
                old_entry_raw = _strip_hash_prefix(entries[idx])
                old_hash = _hash_content(old_entry_raw)
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = prefixed_new
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return {
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content or remove other entries first."
                    ),
                }

            entries[idx] = prefixed_new
            self._set_entries(target, entries)
            self.save_to_disk(target, action="replace", entry_id=new_hash,
                              content=new_content, parent_id=old_hash)

        return self._success_response(target, "Entry replaced.", parent_id=old_hash,
                                      entry_id=new_hash, replaces_id=old_hash)

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        use_tree = bool(self.tree_memory_path)
        with self._file_lock(self._path_for(target, use_tree=use_tree)):
            self._reload_target(target)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            removed_entry = entries[idx]
            removed_hash = _hash_content(_strip_hash_prefix(removed_entry))
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target, action="remove", entry_id=_hash_content(f"remove:{removed_hash}:{int(time.time())}"),
                              content="", parent_id=removed_hash)

        return self._success_response(target, "Entry removed.", removed_id=removed_hash)

    # -------------------------------------------------------------------
    # Phase 3: DAG-based Contradiction Detection, Merge, History
    # -------------------------------------------------------------------

    def _build_dag(self, target: str) -> dict:
        """Build DAG from JSONL: {id: {'content':..., 'parent_id':..., 'action':...}}."""
        use_tree = bool(self.tree_memory_path)
        jsonl_path = self._jsonl_path_for(target, use_tree=use_tree)
        entries_data = _read_jsonl(jsonl_path)
        if not entries_data:
            return {}

        dag = {}
        for entry in entries_data:
            dag[entry["id"]] = {
                "content": entry.get("content", ""),
                "parent_id": entry.get("parent_id"),
                "action": entry.get("action", ""),
            }
        return dag

    def get_contradictions(self) -> list[dict]:
        """Detect entries sharing MRCA with divergent content.

        Builds DAG from JSONL, finds pairs of active entries that share
        the same parent_id but have meaningfully different content.
        Returns list of contradiction dicts with mcra, entries, and contents.
        """
        contradictions = []
        for target in ("memory", "user", "lessons"):
            dag = self._build_dag(target)
            if not dag:
                continue

            # Group active entries by parent_id
            by_parent: dict[str, list[tuple[str, str]]] = {}
            inactive = set()
            # Collect removes first — entries removed by parent_id reference
            removed_by_parent = set()
            for eid, info in dag.items():
                if info.get("action") in ("remove", "merged"):
                    inactive.add(eid)
                    pid = info.get("parent_id")
                    if pid:
                        removed_by_parent.add(pid)

            # Entries superseded by a remove/merged entry are also inactive
            inactive.update(removed_by_parent)

            # Only consider active entries that have a parent
            for eid, info in dag.items():
                if eid in inactive:
                    continue
                pid = info.get("parent_id")
                if pid:
                    # MRCA can be superseded — we still need to report
                    # the divergence of its active children
                    by_parent.setdefault(pid, []).append((eid, info["content"]))

            # Find siblings with divergent content
            for parent_id, siblings in by_parent.items():
                if len(siblings) < 2:
                    continue
                # Check if contents diverge (not identical, not one a substring of another)
                contents = [content for _, content in siblings]
                unique_contents = set(contents)
                if len(unique_contents) <= 1:
                    continue

                # They diverge — report contradiction
                contradictions.append({
                    "target": target,
                    "mrca": parent_id,
                    "entries": [
                        {"id": eid, "content": content[:120] + ("..." if len(content) > 120 else "")}
                        for eid, content in siblings
                    ],
                })

        return contradictions

    def merge(self, target: str, ids: list[str], resolution: str) -> dict:
        """Merge conflicting entries by creating a new entry parented to MRCA.

        Marks the merged entries as 'merged' in JSONL and creates a new
        consolidated entry with parent_id = MRCA.
        """
        if not ids or len(ids) < 2:
            return {"success": False, "error": "At least 2 entry IDs are required to merge."}
        if not resolution or not resolution.strip():
            return {"success": False, "error": "Resolution content cannot be empty."}

        resolution = resolution.strip()
        scan_error = _scan_memory_content(resolution)
        if scan_error:
            return {"success": False, "error": scan_error}

        dag = self._build_dag(target)
        if not dag:
            return {"success": False, "error": "No entries found in DAG."}

        # Validate all ids exist
        for eid in ids:
            if eid not in dag:
                return {"success": False, "error": f"Entry '{eid}' not found in DAG."}

        # Find MRCA: the parent all entries share (must be the same)
        parents = {dag[eid].get("parent_id") for eid in ids}
        if len(parents) != 1:
            return {
                "success": False,
                "error": f"Entries do not share a single MRCA. Found parents: {parents}.",
            }

        mrca = parents.pop()
        if mrca is None:
            return {"success": False, "error": "Entries have no common ancestor (parent_id is None)."}

        # Verify the entries actually diverge
        contents = {dag[eid]["content"] for eid in ids}
        if len(contents) <= 1:
            return {"success": False, "error": "Entries are not divergent (identical content)."}

        use_tree = bool(self.tree_memory_path)
        with self._file_lock(self._path_for(target, use_tree=use_tree)):
            self._reload_target(target)

            new_hash = _hash_content(resolution)
            ts = int(time.time())

            # 1. Mark merged entries as 'merged' in JSONL
            merged_entries = []
            for eid in ids:
                merged_entries.append({
                    "id": eid,
                    "content": dag[eid]["content"],
                    "parent_id": mrca,
                    "action": "merged",
                    "ts": ts,
                    "session_id": os.environ.get("HERMES_SESSION_ID", ""),
                    "orphan": False,
                })

            # 2. Create new consolidated entry
            new_entry = [{
                "id": new_hash,
                "content": resolution,
                "parent_id": mrca,
                "action": "merge",
                "ts": ts,
                "session_id": os.environ.get("HERMES_SESSION_ID", ""),
                "orphan": False,
            }]

            _append_jsonl(self._jsonl_path_for(target, use_tree=use_tree), merged_entries + new_entry)
            self._rebuild_md_from_jsonl(target, use_tree=use_tree)

        return self._success_response(
            target,
            f"Merged {len(ids)} contradictory entries (MRCA: {mrca}).",
            entry_id=new_hash,
            parent_id=mrca,
            merged_ids=ids,
        )

    def history(self, target: str, entry_id: str) -> dict:
        """Return the full lineage chain for an entry: ancestors up to root.

        Walks parent_id chain in the JSONL DAG, returning ordered list
        from oldest (root) to newest (the entry itself).
        """
        if not entry_id:
            return {"success": False, "error": "entry_id is required for 'history' action."}

        dag = self._build_dag(target)
        if not dag:
            return {"success": False, "error": "No DAG entries found."}

        if entry_id not in dag:
            return {"success": False, "error": f"Entry '{entry_id}' not found in DAG."}

        # Walk parent chain
        lineage = []
        current = entry_id
        visited = set()
        max_depth = 50  # safety limit

        while current and len(lineage) < max_depth:
            if current in visited:
                lineage.append({"id": current, "error": "cycle_detected"})
                break
            visited.add(current)

            info = dag.get(current)
            if info is None:
                lineage.append({"id": current, "error": "missing_from_dag"})
                break

            lineage.append({
                "id": current,
                "content": info.get("content", "")[:120] + ("..." if len(info.get("content", "")) > 120 else ""),
                "action": info.get("action", ""),
                "parent_id": info.get("parent_id"),
            })

            current = info.get("parent_id")

        # Reverse so oldest (root) is first
        lineage.reverse()

        return {
            "success": True,
            "target": target,
            "entry_id": entry_id,
            "lineage": lineage,
            "depth": len(lineage),
        }

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        When tree_memory_path is set (dual mode):
          Returns global block (read-only) + tree block (read-write).

        When tree_memory_path is None (standard mode):
          Returns the global block only (behavior unchanged).

        Returns None if the snapshot is empty (no entries at load time).
        """
        tree_block = self._system_prompt_snapshot.get(target, "")
        global_block = self._global_snapshot.get(target, "")

        if self.tree_memory_path:
            # Dual mode: show both blocks
            parts: List[str] = []
            if global_block:
                parts.append(self._wrap_global_label(global_block))
            if tree_block:
                parts.append(self._wrap_tree_label(tree_block, target))
            result = "\n\n".join(parts) if parts else None
            return result
        else:
            # Standard mode: show only the main block
            return tree_block if tree_block else None

    def _wrap_global_label(self, block: str) -> str:
        """Add a (global, read-only) label to a memory block."""
        separator = "─" * 46
        label = f"[GLOBAL MEMORY — read-only reference, do not modify]"
        return f"{separator}\n{label}\n{block}"

    def _wrap_tree_label(self, block: str, target: str) -> str:
        """Add a (tree, read-write) label to a memory block."""
        separator = "─" * 46
        if target == "user":
            label = f"[TREE: USER PROFILE — your read-write memory for this tree]"
        elif target == "lessons":
            label = f"[TREE: LECCIONES — your read-write memory for this tree (stored as memory_private.md)]"
        else:
            label = f"[TREE: MEMORY — your read-write memory for this tree]"
        return f"{separator}\n{label}\n{block}"

    # -- Internal helpers --

    def _success_response(self, target: str, message: str = None, parent_id: str = None, **extra) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        resp = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        if parent_id:
            resp["parent_id"] = parent_id
        resp.update(extra)
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        elif target == "lessons":
            header = f"LECCIONES (IA lessons without PII) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp_path, path)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def memory_tool(
    action: str,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    parent: str = None,
    ids: list = None,
    resolution: str = None,
    entry_id: str = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user", "lessons"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory', 'user', or 'lessons'.", success=False)

    if action == "add":
        if not content:
            return tool_error("Content is required for 'add' action.", success=False)
        result = store.add(target, content)

    elif action == "replace":
        if not old_text:
            return tool_error("old_text is required for 'replace' action.", success=False)
        if not content:
            return tool_error("content is required for 'replace' action.", success=False)
        result = store.replace(target, old_text, content, parent=parent)

    elif action == "remove":
        if not old_text:
            return tool_error("old_text is required for 'remove' action.", success=False)
        result = store.remove(target, old_text)

    elif action == "merge":
        if not ids or not resolution:
            return tool_error("ids (array) and resolution (string) are required for 'merge' action.", success=False)
        result = store.merge(target, ids, resolution)

    elif action == "history":
        if not entry_id:
            return tool_error("entry_id is required for 'history' action.", success=False)
        result = store.history(target, entry_id)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove, merge, history", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable information to persistent memory that survives across sessions. "
        "Memory is injected into future turns, so keep it compact and focused on facts "
        "that will still matter later.\n\n"
        "WHEN TO SAVE (do this proactively, don't wait to be asked):\n"
        "- User corrects you or says 'remember this' / 'don't do that again'\n"
        "- User shares a preference, habit, or personal detail (name, role, timezone, coding style)\n"
        "- You discover something about the environment (OS, installed tools, project structure)\n"
        "- You learn a convention, API quirk, or workflow specific to this user's setup\n"
        "- You identify a stable fact that will be useful again in future sessions\n\n"
        "PRIORITY: User preferences and corrections > environment facts > procedural knowledge. "
        "The most valuable memory prevents the user from having to repeat themselves.\n\n"
        "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
        "state to memory; use session_search to recall those from past transcripts.\n"
        "If you've discovered a new way to do something, solved a problem that could be "
        "necessary later, save it as a skill with the skill tool.\n\n"
        "TWO TARGETS:\n"
        "- 'user': who the user is -- name, role, preferences, communication style, pet peeves\n"
        "- 'memory': your notes -- environment facts, project conventions, tool quirks, lessons learned\n\n"
        "ACTIONS: add (new entry), replace (update existing -- old_text identifies it), "
        "remove (delete -- old_text identifies it), "
        "merge (resolve conflicting entries -- provide ids array and resolution text), "
        "history (show lineage chain for an entry -- provide entry_id).\n\n"
        "SKIP: trivial/obvious info, things easily re-discovered, raw data dumps, and temporary task state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "merge", "history"],
                "description": "The action to perform."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user", "lessons"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile, 'lessons' for PII-free IA improvement lessons."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace'."
            },
            "old_text": {
                "type": "string",
                "description": "Short unique substring identifying the entry to replace or remove."
            },
            "parent": {
                "type": "string",
                "description": "Hash ID of the entry being replaced. Auto-detected if omitted."
            },
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entry hash IDs to merge (for 'merge' action)."
            },
            "resolution": {
                "type": "string",
                "description": "Resolution text for the merged entry (for 'merge' action)."
            },
            "entry_id": {
                "type": "string",
                "description": "Hash ID of the entry to show history for (for 'history' action)."
            },
        },
        "required": ["action", "target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        parent=args.get("parent"),
        ids=args.get("ids"),
        resolution=args.get("resolution"),
        entry_id=args.get("entry_id"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)




