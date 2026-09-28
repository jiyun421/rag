"""Step 7: Reranker (pretrained cross-encoder, fine-tuning 없음).

지금까지 쓴 BM25/Dense는 전부 "bi-encoder" 방식이다: 질문과 문서를 각각 따로
벡터로 만든 다음 벡터끼리 비교한다. 그래서 문서 벡터를 미리 계산해두고 11,865개
전체를 빠르게 훑을 수 있다 (Dense가 그랬던 것처럼).

Cross-encoder는 반대다: 질문과 문서를 "한 문장처럼 붙여서" 하나의 트랜스포머에
같이 넣는다. 그러면 모델이 질문 토큰과 문서 토큰을 서로 직접 attention으로 비교할
수 있어서 미묘한 관계(부정, 특정 숫자가 맞는 항목인지 등)를 훨씬 더 정확히 잡아낸다.
대신 후보 하나마다 forward pass를 새로 해야 해서 11,865개 전체에는 못 쓰고,
이미 추려진 소수 후보(Top20 안팎)를 다시 정렬하는 데만 쓴다.
  - Bi-encoder(Dense)  : 넓은 후보 중에서 빠르게 추린다 (재현율 담당)
  - Cross-encoder(이번) : 추려진 후보를 정확하게 다시 줄세운다 (정밀도 담당)

후보 pool은 BM25 top10 ∪ Dense top10 (합집합, 중복 제거 시 보통 11~20개)을 쓴다.
이렇게 하면 top20을 새로 뽑기 위해 BM25/Dense 인덱스를 다시 만들 필요가 없고,
이미 저장된 top10 파일만 재사용하면 된다.

실행: python src/rerank.py
"""
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RUNS = ROOT / "runs/rerank_test200"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 약 90MB, CPU로 충분


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def merge_by_qid(*result_lists: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for results in result_lists:
        for r in results:
            merged.setdefault(r["qid"], {"qid": r["qid"], "gold_unit_ids": r["gold_unit_ids"]}).update(r)
    return merged


def candidate_pool(*id_lists: list[str]) -> list[str]:
    """여러 방법의 순위 리스트를 합집합(순서 유지, 중복 제거)한다."""
    seen: list[str] = []
    for ids in id_lists:
        for uid in ids:
            if uid not in seen:
                seen.append(uid)
    return seen


def rerank(model: CrossEncoder, question: str, pool: list[str], text_of: dict[str, str]) -> list[tuple[str, float]]:
    pairs = [(question, text_of[uid]) for uid in pool]
    scores = model.predict(pairs)
    return sorted(zip(pool, scores), key=lambda x: -x[1])


def print_comparison(question: str, gold: set[str], text_of: dict[str, str],
                      bm25: list[str], dense: list[str], hybrid: list[str],
                      reranked: list[tuple[str, float]], k: int = 5) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"{'Rank':<5} {'BM25':<28} {'Dense':<28} {'Hybrid':<28} {'Reranker':<28}")
    rerank_ids = [uid for uid, _ in reranked]
    row = lambda uid: (uid + " *" if uid and uid in gold else uid)
    for rank in range(k):
        cells = [lst[rank] if rank < len(lst) else "" for lst in (bm25, dense, hybrid, rerank_ids)]
        print(f"{rank + 1:<5} " + " ".join(f"{row(c):<28}" for c in cells))
    print("(* = gold evidence)\n")
    print(f"Top {k} Retrieved Evidence (Reranker)\n")
    for rank, (uid, score) in enumerate(reranked[:k], 1):
        mark = " [GOLD]" if uid in gold else ""
        print(f"Rank {rank} (ce_score={score:.3f}){mark}\n{text_of[uid]}\n")


def main() -> None:
    units = load_jsonl(PROC / "corpus.jsonl")
    text_of = {u["unit_id"]: u["text"] for u in units}
    question_of = {q["qid"]: q["question"] for q in load_jsonl(PROC / "questions_test200.jsonl")}

    bm25_results = load_jsonl(ROOT / "runs/bm25_test200/bm25_top10.jsonl")
    dense_results = load_jsonl(ROOT / "runs/dense_test200/dense_top10.jsonl")
    hybrid_results = load_jsonl(ROOT / "runs/hybrid_test200/hybrid_top10.jsonl")
    merged = merge_by_qid(bm25_results, dense_results, hybrid_results)

    model = CrossEncoder(MODEL_NAME)

    results = []
    for i, (qid, r) in enumerate(merged.items()):
        question = question_of[qid]
        bm25_in = [x["unit_id"] for x in r["bm25_in_page_top10"]]
        bm25_pool = [x["unit_id"] for x in r["bm25_pooled_top10"]]
        dense_in = [x["unit_id"] for x in r["dense_in_page_top10"]]
        dense_pool = [x["unit_id"] for x in r["dense_pooled_top10"]]
        hybrid_in = [x["unit_id"] for x in r["hybrid_in_page_top10"]]
        hybrid_pool = [x["unit_id"] for x in r["hybrid_pooled_top10"]]

        pool_in = candidate_pool(bm25_in, dense_in)
        pool_pool = candidate_pool(bm25_pool, dense_pool)
        rerank_in = rerank(model, question, pool_in, text_of)
        rerank_pool = rerank(model, question, pool_pool, text_of)

        results.append({"qid": qid, "gold_unit_ids": r["gold_unit_ids"],
                        "reranker_in_page_top10": [{"unit_id": u, "score": float(s)} for u, s in rerank_in[:10]],
                        "reranker_pooled_top10": [{"unit_id": u, "score": float(s)} for u, s in rerank_pool[:10]]})

        if i < 2:
            gold = set(r["gold_unit_ids"])
            print("=" * 115, f"\n[in_page 설정, 후보 {len(pool_in)}개]")
            print_comparison(question, gold, text_of, bm25_in, dense_in, hybrid_in, rerank_in)
            print("-" * 115, f"\n[pooled 설정, 후보 {len(pool_pool)}개]")
            print_comparison(question, gold, text_of, bm25_pool, dense_pool, hybrid_pool, rerank_pool)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "reranker_top10.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"\n저장 완료: {out} ({len(results)} questions)")


if __name__ == "__main__":
    main()
