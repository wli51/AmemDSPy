"""
memory_system.py

Core memory system.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Literal, FrozenSet

from .retrievers import ChromaRetriever
from .memory_note import MemoryNote
from .memory_index import MemoryIndex


class AgenticMemorySystem:
    """
    Core memory system that manages memory.    
    Functionalities:
    - Memory creation, retrieval, update, and deletion (CRUD operations)
    - Embedding-based semantic search
    - Metadata management
    - Expandability for integrating LLM-based memory processing
    """
    
    def __init__(
        self, 
        retriever: Optional[ChromaRetriever] = None,
        **kwargs
    ):
        """
        Initialize the memory system.
        
        :param collection_name: Name of the ChromaDB collection
        :param reset_collection: If True, reset the collection on init
        :param kwargs: Additional args for ChromaRetriever
        """
        self.memories: Dict[str, MemoryNote] = {}
        # keyword index to memory ID mapping cache for faster lookup
        self.kw_index = MemoryIndex()
        # self._index_to_id: Dict[FrozenSet[Tuple[str, str]], str] = {}
        # self._kv_to_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        # Initialize ChromaDB retriever
        if retriever is None:
            try:
                self.retriever = ChromaRetriever(**kwargs)
                self.retriever.client.reset()
            except Exception as e:
                raise RuntimeError(
                    f"Error during initializing ChromaDB retriever: {e}")

        elif isinstance(retriever, ChromaRetriever):
            self.retriever = retriever
        else:
            raise TypeError(
                "retriever must be a ChromaRetriever instance, "
                f"got {type(retriever)}"
            )
        
        # for future LLM integration
        self._llm_processor = None
        
        # Load existing memories from the retriever
        self._load_existing_memories()

    def _load_existing_memories(self):
        """
        Load all existing memories from the retriever into self.memories.
        Called during initialization to sync in-memory state with ChromaDB.
        """
        try:
            all_data = self.retriever.collection.get(include=["metadatas", "documents"])
            if not all_data or not all_data.get("ids"):
                return

            if all_data.get("metadatas"):
                all_data["metadatas"] = self.retriever._deserialize_metadatas(
                    [all_data["metadatas"]]
                )[0]

            # bucket by exact canonical index to collapse legacy duplicates
            buckets: Dict[FrozenSet[Tuple[str, str]], List[str]] = {}

            for doc_id, content, metadata in zip(
                all_data["ids"], all_data["documents"], all_data["metadatas"]
            ):
                if "content" not in metadata:
                    metadata["content"] = content
                if "id" not in metadata:
                    metadata["id"] = doc_id

                note = MemoryNote.deserialize_from_storage(metadata)
                self.memories[doc_id] = note

                extras = getattr(note, "extras", None)
                self.kw_index.add_bucket_candidate(doc_id, extras)

            def _resolve_dup(ids: List[str]) -> str:
                def _score(_id: str):
                    n = self.memories[_id]
                    ts = getattr(n, "timestamp", "") or ""
                    la = getattr(n, "last_accessed", None)
                    la_ts = int(la.timestamp()) if la and hasattr(la, "timestamp") else 0
                    rc = getattr(n, "retrieval_count", 0) or 0
                    return (ts, la_ts, rc)
                return sorted(ids, key=_score, reverse=True)[0]

            self.kw_index.finalize_buckets(_resolve_dup)

        except AttributeError:
            pass
        except Exception as e:
            import warnings
            warnings.warn(
                f"Failed to load memories or build index cache: {e}", RuntimeWarning
            )
    
    def read(self, memory_id: str) -> Optional[MemoryNote]:
        memory = self.memories.get(memory_id)
        if memory:
            memory.last_accessed = datetime.now(timezone.utc)
            memory.retrieval_count += 1
        return memory
    
    def semantic_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Pure embedding-based semantic search via retriever.
        Returns a list of dicts:
          - if memory is known locally: MemoryNote.model_dump()
          - otherwise: {"id": <id>, "content": <text>} as a lightweight stub
        """
        results_list: List[Dict[str, Any]] = []
        results = self.retriever.search(query, k)
        if results and results.get("ids"):
            for id_, content, distance in zip(
                results["ids"][0], results["documents"][0], results["distances"][0]
            ):
                if distance <= threshold:
                    if id_ in self.memories:
                        results_list.append(self.memories[id_].model_dump())
                    else:
                        results_list.append({"id": id_, "content": content})
        return results_list[:k]
    
    def filter_by_index(
        self,
        index: Dict[str, str],
        mode: Literal["exact", "subset", "any"] = "exact",
        return_ids: bool = False,
    ) -> List[Any]:
        """
        Keyword-index filter ONLY (no embeddings).
        - exact: full-key match -> at most one ID
        - subset: partial match -> intersect per (k,v)
        Returns MemoryNote objects by default, or IDs if return_ids=True.
        """
        if mode not in {"exact", "subset", "any"}:
            raise ValueError("mode must be 'exact', 'subset' or 'any'")

        if mode == "exact":
            ids = self.kw_index.ids_by_exact_index(index)
        elif mode == "subset":
            ids = self.kw_index.ids_by_partial_index(index)
        elif mode == "any":
            ids = self.kw_index.ids_by_any_index(index)
        else:
            # should not reach here
            pass

        if return_ids:
            return ids

        # map to MemoryNotes (skip any dangling ids defensively)
        return [self.memories[mid].model_dump() for mid in ids if mid in self.memories]
    
    def upsert(self, content: str, **index: str) -> str:
        """
        - Indexed: upsert(content, k1="v1", k2="v2", ...)
                    updates existing or creates with deterministic id
        - Non-indexed: upsert(content)  # creates a new free-form note (extras = {})
        """
        if index:
            norm = MemoryIndex.normalize_index(index)

            # update path
            memo_id = self.kw_index.get_id_by_index(norm)
            if memo_id:
                note = self.memories[memo_id]
                note.update(content=content, extras=norm)
                self.retriever.collection.update(
                    ids=[memo_id],
                    documents=[note.content],
                    metadatas=[note.serialize_for_storage()],
                )
                # caches already point at this id for this index
                return memo_id

            # create path (deterministic by default)
            memo_id = MemoryIndex.deterministic_id_for_index(norm)
            note = MemoryNote(id=memo_id, content=content, extras=norm)
            memo_id = note.id
            self.memories[memo_id] = note
            self.kw_index.register_index(memo_id, norm)

            self.retriever.add_document(
                document=note.content,
                metadata=note.serialize_for_storage(),
                doc_id=memo_id,
            )
            return memo_id

        # Non-indexed: always create new, extras = {}
        note = MemoryNote(content=content, extras={})
        memo_id = note.id
        self.memories[memo_id] = note
        self.retriever.add_document(
            document=note.content,
            metadata=note.serialize_for_storage(),
            doc_id=memo_id,
        )
        return memo_id
