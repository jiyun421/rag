"""Step 3: BM25 baseline.

두 가지 설정을 모두 돌린다.
  - in_page : 질문이 속한 문서(페이지) 안의 units만 놓고 검색 (원 FinQA retriever와 같은 설정)
  - pooled  : 380개 문서의 units 11,865개 전체를 놓고 검색 (본 실험)

실행: python src/bm25_retrieve.py
"""
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RUNS = ROOT / "runs/bm25_test200"

# 숫자($ 5829, 8.1 같은)를 통째로 살리는 토크나이저. 금융 문서는 숫자 하나가 곧 evidence라
# "8.1"을 "8"과 "1"로 쪼개면 BM25 IDF 계산이 왜곡된다.
TOKEN = re.compile(r"\$|%|\d+\.\d+|\d+|[a-z]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build_index(units: list[dict]) -> tuple[BM25Okapi, list[str]]:
    ids = [u["unit_id"] for u in units]
    bm25 = BM25Okapi([tokenize(u["text"]) for u in units])
    return bm25, ids


def topk(bm25: BM25Okapi, ids: list[str], query: str, k: int) -> list[tuple[str, float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
    return ranked[:k]


def print_result(q: dict, text_of: dict[str, str], ranked: list[tuple[str, float]], k: int = 5) -> None:
    print(f"\nQuestion:\n{q['question']}\n")
    print(f"Top {k} Retrieved Evidence\n")
    for rank, (uid, score) in enumerate(ranked[:k], 1):
        mark = " [GOLD]" if uid in q["gold_unit_ids"] else ""
        print(f"Rank {rank} (score={score:.2f}){mark}\n{text_of[uid]}\n")


def main() -> None:
    units = load_jsonl(PROC / "corpus.jsonl")
    questions = load_jsonl(PROC / "questions_test200.jsonl")
    text_of = {u["unit_id"]: u["text"] for u in units}
    by_doc: dict[str, list[dict]] = {}
    for u in units:
        by_doc.setdefault(u["doc_id"], []).append(u)

    pooled_bm25, pooled_ids = build_index(units)
    page_index_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}

    results = []
    for i, q in enumerate(questions):
        if q["doc_id"] not in page_index_cache:
            page_index_cache[q["doc_id"]] = build_index(by_doc[q["doc_id"]])
        page_bm25, page_ids = page_index_cache[q["doc_id"]]

        in_page = topk(page_bm25, page_ids, q["question"], 10)
        pooled = topk(pooled_bm25, pooled_ids, q["question"], 10)
        results.append({"qid": q["qid"], "gold_unit_ids": q["gold_unit_ids"],
                        "bm25_in_page_top10": [{"unit_id": u, "score": s} for u, s in in_page],
                        "bm25_pooled_top10": [{"unit_id": u, "score": s} for u, s in pooled]})

        if i < 2:  # 처음 2개 질문만 화면에 보여준다 (나머지는 파일로 저장)
            print("=" * 70, f"\n[in_page 설정, 후보 {len(page_ids)}개]")
            print_result(q, text_of, in_page)
            print("-" * 70, f"\n[pooled 설정, 후보 {len(pooled_ids)}개]")
            print_result(q, text_of, pooled)

    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "bm25_top10.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"\n저장 완료: {out} ({len(results)} questions)")


if __name__ == "__main__":
    main()
