"""Step 5: Dense retrieval (embedding + cosine similarity, numpy 기반).

BM25와 달리 dense는 corpus 전체 통계(IDF)에 의존하지 않는다. 그래서 in_page용
인덱스를 따로 만들 필요 없이, 임베딩 행렬 하나를 만들고 후보 범위만 좁히면 된다.

실행: python src/dense_retrieve.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RUNS = ROOT / "runs/dense_test200"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_CACHE = PROC / "embeddings_minilm.npy"  # .gitignore의 *.npy에 이미 걸려 있음


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def normalize(vecs: np.ndarray) -> np.ndarray:
    """L2 정규화. 벡터 길이를 1로 맞추면 dot product == cosine similarity가 된다."""
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_corpus(model: SentenceTransformer, units: list[dict]) -> np.ndarray:
    """전체 임베딩은 한 번만 계산하고 캐시한다 (재실행 때마다 몇 분씩 걸리지 않도록)."""
    if EMB_CACHE.exists():
        return np.load(EMB_CACHE)
    emb = normalize(model.encode([u["text"] for u in units], batch_size=64,
                                  show_progress_bar=True, convert_to_numpy=True))
    np.save(EMB_CACHE, emb)
    return emb


def topk(scores: np.ndarray, ids: list[str], k: int) -> list[tuple[str, float]]:
    order = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in order]


def print_result(question: str, gold: set[str], text_of: dict[str, str],
                  ranked: list[tuple[str, float]], k: int = 5) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"Top {k} Retrieved Evidence\n")
    for rank, (uid, score) in enumerate(ranked[:k], 1):
        mark = " [GOLD]" if uid in gold else ""
        print(f"Rank {rank} (cos={score:.3f}){mark}\n{text_of[uid]}\n")


def main() -> None:
    units = load_jsonl(PROC / "corpus.jsonl")
    questions = load_jsonl(PROC / "questions_test200.jsonl")
    text_of = {u["unit_id"]: u["text"] for u in units}
    unit_ids = [u["unit_id"] for u in units]
    doc_of = {u["unit_id"]: u["doc_id"] for u in units}

    model = SentenceTransformer(MODEL_NAME)
    corpus_emb = embed_corpus(model, units)  # (11865, 384), 이미 정규화됨
    q_emb = normalize(model.encode([q["question"] for q in questions], convert_to_numpy=True))

    # doc_id -> corpus_emb 행 인덱스 목록 (in_page 검색용 부분집합)
    idx_by_doc: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        idx_by_doc.setdefault(u["doc_id"], []).append(i)

    results = []
    for i, q in enumerate(questions):
        sims = corpus_emb @ q_emb[i]  # (11865,) 정규화된 벡터끼리 내적 = cosine similarity

        page_idx = idx_by_doc[q["doc_id"]]
        page_scores = sims[page_idx]
        in_page = topk(page_scores, [unit_ids[j] for j in page_idx], 10)
        pooled = topk(sims, unit_ids, 10)

        results.append({"qid": q["qid"], "gold_unit_ids": q["gold_unit_ids"],
                        "dense_in_page_top10": [{"unit_id": u, "score": s} for u, s in in_page],
                        "dense_pooled_top10": [{"unit_id": u, "score": s} for u, s in pooled]})

        if i < 2:
            gold = set(q["gold_unit_ids"])
            print("=" * 70, f"\n[in_page 설정, 후보 {len(page_idx)}개]")
            print_result(q["question"], gold, text_of, in_page)
            print("-" * 70, "\n[pooled 설정, 후보 11865개]")
            print_result(q["question"], gold, text_of, pooled)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "dense_top10.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"\n저장 완료: {out} ({len(results)} questions)")


if __name__ == "__main__":
    main()
