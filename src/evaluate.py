"""Step 4: Retrieval evaluation metrics (Recall@k, MRR, NDCG@k).

BM25뿐 아니라 이후 Dense/Hybrid/Reranker 결과도 같은 형식(qid, gold_unit_ids, top10)이면
그대로 채점할 수 있도록 만든다.

실행: python src/evaluate.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """질문 1개 기준: gold 중 상위 k개 안에 들어온 비율. gold가 여러 개면 부분 점수가 나온다."""
    found = len(set(ranked_ids[:k]) & gold)
    return found / len(gold)


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    """gold를 처음 맞춘 순위의 역수. 1등에서 맞추면 1.0, 5등에서 맞추면 0.2. 못 찾으면 0."""
    for rank, uid in enumerate(ranked_ids, 1):
        if uid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """gold=1, 아니면 0인 이진 관련도로 계산. gold를 앞쪽 순위에 몰아넣을수록 1.0에 가까워진다."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, uid in enumerate(ranked_ids[:k], 1) if uid in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(results: list[dict], field: str) -> dict[str, float]:
    """results: qid, gold_unit_ids, <field>(=score 붙은 랭킹 리스트)를 담은 딕셔너리 목록."""
    r5 = r10 = mrr = ndcg10 = 0.0
    n = len(results)
    for r in results:
        ranked = [x["unit_id"] for x in r[field]]
        gold = set(r["gold_unit_ids"])
        r5 += recall_at_k(ranked, gold, 5)
        r10 += recall_at_k(ranked, gold, 10)
        mrr += reciprocal_rank(ranked, gold)
        ndcg10 += ndcg_at_k(ranked, gold, 10)
    return {"Recall@5": r5 / n, "Recall@10": r10 / n, "MRR": mrr / n, "NDCG@10": ndcg10 / n}


def print_table(rows: dict[str, dict[str, float]]) -> None:
    cols = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]
    print(f"| {'Method':<16} | " + " | ".join(f"{c:>9}" for c in cols) + " |")
    print(f"|{'-'*18}|" + "".join(f"{'-'*11}|" for _ in cols))
    for name, m in rows.items():
        print(f"| {name:<16} | " + " | ".join(f"{m[c]:9.3f}" for c in cols) + " |")


def worked_example(results: list[dict], field: str) -> None:
    """11번 항목에서 설명한 계산을 실제 데이터 한 건으로 재현해 보여준다."""
    r = next(x for x in results if x["qid"] == "INTC/2015/page_41.pdf-4")
    ranked = [x["unit_id"] for x in r[field]]
    gold = set(r["gold_unit_ids"])
    print(f"\n[검산] {r['qid']} | gold={sorted(gold)}")
    for rank, uid in enumerate(ranked, 1):
        print(f"  Rank {rank}: {uid}" + ("  <- gold" if uid in gold else ""))
    print(f"  Recall@5={recall_at_k(ranked, gold, 5):.2f}  Recall@10={recall_at_k(ranked, gold, 10):.2f}")


def merge_by_qid(*result_lists: list[dict]) -> list[dict]:
    """method별로 따로 저장된 run 파일들(runs/<method>_test200/*_top10.jsonl)을
    qid 기준으로 합친다. 같은 질문에 대해 여러 방법의 top10을 한 번에 채점하기 위함."""
    merged: dict[str, dict] = {}
    for results in result_lists:
        for r in results:
            merged.setdefault(r["qid"], {"qid": r["qid"], "gold_unit_ids": r["gold_unit_ids"]}).update(r)
    return list(merged.values())


def main() -> None:
    bm25 = load_jsonl(ROOT / "runs/bm25_test200/bm25_top10.jsonl")
    dense = load_jsonl(ROOT / "runs/dense_test200/dense_top10.jsonl")
    results = merge_by_qid(bm25, dense)

    table = {
        "BM25 (in_page)": evaluate(results, "bm25_in_page_top10"),
        "BM25 (pooled)": evaluate(results, "bm25_pooled_top10"),
        "Dense (in_page)": evaluate(results, "dense_in_page_top10"),
        "Dense (pooled)": evaluate(results, "dense_pooled_top10"),
    }
    print_table(table)
    worked_example(results, "bm25_pooled_top10")

    out = ROOT / "runs/metrics.json"
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
