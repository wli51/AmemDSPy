# memory_index.py

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set, FrozenSet, Callable


class MemoryIndex:
    """
    Runtime memory index (cache) for keyword-indexed MemoryNotes.

    - Exact index (as frozenset of (k,v)) -> single memory id
    - Inverted index: (k,v) -> set(ids) for partial/subset matches
    - Bucket accumulation -> finalize with a resolver to collapse legacy dups

    Backend-agnostic: caller decides how to fetch notes and what a "better" id is
    when collapsing duplicates (via resolver).
    """

    def __init__(self) -> None:
        self._index_to_id: Dict[FrozenSet[Tuple[str, str]], str] = {}
        self._kv_to_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # transient during load
        self._buckets: Dict[FrozenSet[Tuple[str, str]], List[str]] = {}

    # Static index helpers

    @staticmethod
    def normalize_index(index: Dict[str, Any]) -> Dict[str, str]:
        if not isinstance(index, dict) or not index:
            raise ValueError("Keyword index must be a non-empty dict of str->str.")
        out: Dict[str, str] = {}
        for k, v in index.items():
            if not isinstance(k, str):
                raise TypeError("Index keys must be strings.")
            if not isinstance(v, str):
                raise TypeError(f"Index value for '{k}' must be a string.")
            key = k.strip().lower()
            val = v.strip()
            if not key or not val:
                raise ValueError("Index keys/values must be non-empty strings.")
            out[key] = val
        return out

    @staticmethod
    def index_fset(index: Dict[str, str]) -> FrozenSet[Tuple[str, str]]:
        return frozenset(sorted(index.items()))

    @staticmethod
    def deterministic_id_for_index(index: Dict[str, str]) -> str:
        s = json.dumps(index, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        return f"kwidx:{h}"

    @staticmethod
    def is_indexed_extras(extras: Any) -> bool:
        return isinstance(extras, dict) and len(extras) > 0

    # Init time construction helpers to be called by memory system

    def add_bucket_candidate(self, doc_id: str, extras: Dict[str, Any]) -> None:
        """
        Call this inside your load loop for each note that has indexed extras.
        """
        if not self.is_indexed_extras(extras):
            return
        norm = self.normalize_index(extras)
        key = self.index_fset(norm)
        self._buckets.setdefault(key, []).append(doc_id)

    def finalize_buckets(self, resolver: Callable[[List[str]], str]) -> None:
        """
        After the load loop, collapse duplicates and populate caches.

        resolver: given a list of doc_ids for the same index, return the winner id.
                  (e.g., prefer most recently accessed / highest retrieval_count)
        """
        for key_fset, ids in self._buckets.items():
            chosen_id = resolver(ids) if len(ids) > 1 else ids[0]
            self._index_to_id[key_fset] = chosen_id
            for kv in key_fset:
                self._kv_to_ids[kv].add(chosen_id)

        # drop temp buckets
        self._buckets.clear()

    # Runtime query methods

    def get_id_by_index(self, index: Dict[str, Any]) -> Optional[str]:
        norm = self.normalize_index(index)
        key = self.index_fset(norm)
        return self._index_to_id.get(key)
    
    def ids_by_any_index(self, index: Dict[str, Any]) -> List[str]:
        """
        OR-style match: return IDs of notes that share *any* (k, v) pair
        with the provided index.
        e.g. index = {"color": "yellow", "fruit": "banana"} would match notes with
        .extras = {"color": "red", "fruit": "apple"}
        and .extras = {"color": "red", "fruit": "strawberry"},
        """
        norm = self.normalize_index(index)
        candidate_sets = [self._kv_to_ids.get(kv, set()) for kv in norm.items()]
        union_ids: Set[str] = set()
        for s in candidate_sets:
            union_ids |= s
        return list(union_ids)

    def ids_by_partial_index(self, index: Dict[str, Any]) -> List[str]:
        """
        Matches all notes that have at least the supplied (k,v) pairs.
        e.g. index = {"color": "red"} would match notes with
        .extras = {"color": "red", "fruit": "strawberry"}
        and .extras = {"color": "red", "fruit": "apple"}, etc.
        """
        norm = self.normalize_index(index)
        sets: List[Set[str]] = [self._kv_to_ids.get(kv, set()) for kv in norm.items()]
        if not sets:
            return []
        cand = set.intersection(*sets) if len(sets) > 1 else set(sets[0])
        return list(cand)
    
    def ids_by_exact_index(self, index: Dict[str, Any]) -> List[str]:
        """
        Return [id] if the full index matches exactly.
        e.g. index = {"color": "red", "fruit": "strawberry"} would only match
        notes that have precisely .extras = {"color": "red", "fruit": "strawberry"},
        having additional keys unmatched by the index or missing keys supplied
        in the index would result in not matching.
        """
        mid = self.get_id_by_index(index)
        return [mid] if mid else []

    def register_index(self, doc_id: str, index: Dict[str, Any]) -> None:
        """
        After creating/updating an indexed note, update caches.
        """
        norm = self.normalize_index(index)
        key = self.index_fset(norm)
        self._index_to_id[key] = doc_id
        for kv in key:
            self._kv_to_ids[kv].add(doc_id)
