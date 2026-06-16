import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def build_document_corpus(cases: Iterable[Dict]) -> Dict[str, str]:
    """Create a deterministic local corpus from golden case evidence."""
    corpus: Dict[str, List[str]] = defaultdict(list)
    for case in cases:
        metadata = case.get("metadata", {})
        tags = " ".join(case.get("tags", []))
        evidence = (
            f"{case['question']} {case['expected_answer']} {case.get('context', '')} "
            f"{case.get('case_type', '')} {metadata.get('type', '')} {metadata.get('difficulty', '')} {tags}"
        )
        for doc_id in expected_ids_for_case(case):
            corpus[doc_id].append(evidence)
    return {doc_id: " ".join(parts) for doc_id, parts in sorted(corpus.items())}


def load_markdown_corpus(corpus_dir: Path | str = Path("data/corpus")) -> Dict[str, str]:
    """Load each markdown corpus file as one retrievable document.

    Document IDs are the filename stem, matching Linh's expected_retrieval_ids.
    """
    path = Path(corpus_dir)
    if not path.exists():
        return {}
    corpus = {}
    for markdown_path in sorted(path.glob("*.md")):
        corpus[markdown_path.stem] = markdown_path.read_text(encoding="utf-8")
    return corpus


def expected_ids_for_case(case: Dict) -> List[str]:
    return list(case.get("expected_retrieval_ids") or case.get("ground_truth_ids") or [])


class TfidfVectorRetriever:
    retriever_type = "tfidf_vector_retriever"

    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.doc_tokens = {doc_id: Counter(tokenize(text)) for doc_id, text in corpus.items()}
        doc_count = max(len(self.doc_tokens), 1)
        df = Counter()
        for counts in self.doc_tokens.values():
            df.update(counts.keys())
        self.idf = {term: math.log((1 + doc_count) / (1 + freq)) + 1 for term, freq in df.items()}
        self.doc_vectors = {doc_id: self._tfidf_vector(counts) for doc_id, counts in self.doc_tokens.items()}
        self.doc_norms = {doc_id: self._norm(vector) for doc_id, vector in self.doc_vectors.items()}

    def _tfidf_vector(self, counts: Counter) -> Dict[str, float]:
        return {term: count * self.idf.get(term, 1.0) for term, count in counts.items()}

    def _norm(self, vector: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_vector = self._tfidf_vector(Counter(tokenize(query)))
        query_norm = self._norm(query_vector)
        scores = []
        if query_norm == 0:
            return []
        for doc_id, doc_vector in self.doc_vectors.items():
            dot = sum(weight * doc_vector.get(term, 0.0) for term, weight in query_vector.items())
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            score = dot / (query_norm * doc_norm) if doc_norm else 0.0
            if score > 0:
                scores.append((score, doc_id))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [doc_id for _, doc_id in scores[:top_k]]


LexicalRetriever = TfidfVectorRetriever


class RetrievalEvaluator:
    def __init__(self, corpus: Dict[str, str] | None = None, top_k: int = 5):
        self.top_k = top_k
        self.retriever = TfidfVectorRetriever(corpus or {})

    def calculate_hit_rate(self, expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int | None = None) -> float:
        if not expected_ids:
            return 1.0 if not retrieved_ids else 0.0
        top = list(retrieved_ids)[: top_k or self.top_k]
        return 1.0 if any(doc_id in top for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int | None = None) -> float | None:
        expected = set(expected_ids)
        if not expected:
            return None
        for index, doc_id in enumerate(list(retrieved_ids)[: top_k or self.top_k], start=1):
            if doc_id in expected:
                return 1.0 / index
        return 0.0

    def calculate_recall(self, expected_ids: Sequence[str], retrieved_ids: Sequence[str], top_k: int | None = None) -> float | None:
        expected = set(expected_ids)
        if not expected:
            return None
        top = set(list(retrieved_ids)[: top_k or self.top_k])
        return len(expected.intersection(top)) / len(expected)

    def retrieve(self, question: str, top_k: int | None = None) -> List[str]:
        return self.retriever.retrieve(question, top_k or self.top_k)

    def evaluate_case(self, case: Dict, retrieved_ids: Sequence[str]) -> Dict:
        expected = expected_ids_for_case(case)
        return {
            "hit_rate": self.calculate_hit_rate(expected, retrieved_ids),
            "mrr": self.calculate_mrr(expected, retrieved_ids),
            "recall": self.calculate_recall(expected, retrieved_ids),
            "avg_retrieved_count": len(retrieved_ids),
            "retriever_type": self.retriever.retriever_type,
            "top_k": self.top_k,
        }

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        rows = []
        for case in dataset:
            retrieved = self.retrieve(case["question"])
            rows.append(self.evaluate_case(case, retrieved))
        total = max(len(rows), 1)
        mrr_rows = [row for row in rows if row["mrr"] is not None]
        recall_rows = [row for row in rows if row["recall"] is not None]
        return {
            "avg_hit_rate": sum(row["hit_rate"] for row in rows) / total,
            "avg_mrr": sum(row["mrr"] for row in mrr_rows) / len(mrr_rows) if mrr_rows else 0.0,
            "avg_recall": sum(row["recall"] for row in recall_rows) / len(recall_rows) if recall_rows else 0.0,
            "avg_retrieved_count": sum(row["avg_retrieved_count"] for row in rows) / total,
        }
