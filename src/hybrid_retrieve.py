"""Step 6: Hybrid retrieval (BM25 + Dense, Reciprocal Rank Fusion).

RRF는 두 방법의 "점수"가 아니라 "순위"만 이용해 합친다. BM25 점수와 코사인 유사도는
스케일이 완전히 달라서(0~수십 vs 0~1) 점수를 직접 더하면 한쪽이 묻힌다. 순위 기반이면
스케일 문제 없이 두 방법의 장점을 섞을 수 있다.

RRF_score(d) = sum over method m of  1 / (k + rank_m(d))
  - rank_m(d): 방법 m의 결과에서 문서 d의 순위 (1등부터 시작)
  - k=60: 관용적으로 쓰는 상수. 1등과 2등의 점수 차이를 완만하게 만들어,
          상위권 등수 하나 차이로 등수가 요동치지 않게 한다.
  - 한쪽 방법의 top10 안에 아예 없는 문서는 그 방법에서 0으로 취급(더하지 않음).

입력은 이미 저장된 BM25/Dense의 top10 리스트라서, 새로 인덱스를 만들 필요가 없다.

실행: python src/hybrid_retrieve.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RUNS = ROOT / "runs/hybrid_test200"
K_RRF = 60


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def merge_by_qid(*result_lists: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for results in result_lists:
        for r in results:
            merged.setdefault(r["qid"], {"qid": r["qid"], "gold_unit_ids": r["gold_unit_ids"]}).update(r)
    return merged


def rrf_fuse(*ranked_lists: list[str], k: int = K_RRF) -> list[tuple[str, float]]:
    """ranked_lists: 방법별 unit_id 순위 리스트(1등부터). 반환은 RRF 점수 내림차순."""
    score: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, uid in enumerate(ranked, 1):
            score[uid] = score.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def print_comparison(question: str, gold: set[str], text_of: dict[str, str],
                      bm25: list[str], dense: list[str], hybrid: list[tuple[str, float]], k: int = 5) -> None:
    print(f"\nQuestion:\n{question}\n")
    print(f"{'Rank':<5} {'BM25':<28} {'Dense':<28} {'Hybrid':<28}")
    hybrid_ids = [uid for uid, _ in hybrid]
    for rank in range(k):
        b = bm25[rank] if rank < len(bm25) else ""
        d = dense[rank] if rank < len(dense) else ""
        h = hybrid_ids[rank] if rank < len(hybrid_ids) else ""
        row = lambda uid: (uid + " *" if uid and uid in gold else uid)
        print(f"{rank + 1:<5} {row(b):<28} {row(d):<28} {row(h):<28}")
    print("(* = gold evidence)\n")
    print(f"Top {k} Retrieved Evidence (Hybrid)\n")
    for rank, (uid, score) in enumerate(hybrid[:k], 1):
        mark = " [GOLD]" if uid in gold else ""
        print(f"Rank {rank} (rrf={score:.4f}){mark}\n{text_of[uid]}\n")


def main() -> None:
    units = load_jsonl(PROC / "corpus.jsonl")
    text_of = {u["unit_id"]: u["text"] for u in units}

    bm25_results = load_jsonl(ROOT / "runs/bm25_test200/bm25_top10.jsonl")
    dense_results = load_jsonl(ROOT / "runs/dense_test200/dense_top10.jsonl")
    merged = merge_by_qid(bm25_results, dense_results)
    question_of = {q["qid"]: q["question"] for q in load_jsonl(PROC / "questions_test200.jsonl")}

    results = []
    for i, (qid, r) in enumerate(merged.items()):
        bm25_in = [x["unit_id"] for x in r["bm25_in_page_top10"]]
        bm25_pool = [x["unit_id"] for x in r["bm25_pooled_top10"]]
        dense_in = [x["unit_id"] for x in r["dense_in_page_top10"]]
        dense_pool = [x["unit_id"] for x in r["dense_pooled_top10"]]

        hybrid_in = rrf_fuse(bm25_in, dense_in)
        hybrid_pool = rrf_fuse(bm25_pool, dense_pool)

        results.append({"qid": qid, "gold_unit_ids": r["gold_unit_ids"],
                        "hybrid_in_page_top10": [{"unit_id": u, "score": s} for u, s in hybrid_in[:10]],
                        "hybrid_pooled_top10": [{"unit_id": u, "score": s} for u, s in hybrid_pool[:10]]})

        if i < 2:
            gold = set(r["gold_unit_ids"])
            question = question_of[qid]
            print("=" * 90, f"\n[in_page 설정]")
            print_comparison(question, gold, text_of, bm25_in, dense_in, hybrid_in)
            print("-" * 90, f"\n[pooled 설정]")
            print_comparison(question, gold, text_of, bm25_pool, dense_pool, hybrid_pool)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "hybrid_top10.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"\n저장 완료: {out} ({len(results)} questions)")


if __name__ == "__main__":
    main()
