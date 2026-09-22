"""Step 2: FinQA test split -> corpus.jsonl / questions_*.jsonl / manifest.json

실행: python src/prepare_finqa.py
"""
import collections
import hashlib
import json
import random
import re
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"

# 재현성: main 브랜치가 아니라 커밋 해시로 고정한다
COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{COMMIT}/dataset/test.json"
SEED, N_SUBSET = 42, 200


def download() -> tuple[Path, str]:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"finqa_test_{COMMIT[:7]}.json"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def row_to_text(header: list[str], row: list[str]) -> str:
    """표 한 행을 문장으로 바꾼다 (FinQA 원 논문의 템플릿).
    모든 데이터 행에 같은 규칙을 적용하므로 정답 행만 특별 취급하지 않는다."""
    res = header[0] + " " if header[0] else ""
    for h, cell in zip(header[1:], row[1:]):
        res += f"the {row[0]} of {h} is {cell} ; "
    return " ".join(res.split())


def header_to_text(header: list[str]) -> str:
    """헤더 행(table_0) 전용 표현. row_to_text를 헤더에 그대로 적용하면 header를
    자기 자신의 row label처럼 취급해 'the X of X is X' 같은 무의미한 문장이 나온다
    (BM25 실행 결과에서 확인됨). 헤더는 열 이름 나열로만 표현한다."""
    cells = [c for c in header if c]
    return "columns: " + " , ".join(cells) if cells else ""


def build_corpus(examples: list[dict]) -> list[dict]:
    """페이지(filename) 단위로 중복을 제거하고 검색 단위(unit)로 펼친다.
    unit = text 문장 1개 또는 표 행 1개. 한 페이지에 질문이 여러 개라 같은 페이지가 반복 등장한다."""
    units, seen = [], set()
    for e in examples:
        doc = e["filename"]
        if doc in seen:
            continue
        seen.add(doc)
        for i, t in enumerate(e["pre_text"] + e["post_text"]):  # gold의 text_N 인덱스 규칙
            units.append({"unit_id": f"{doc}::text_{i}", "doc_id": doc, "kind": "text", "text": " ".join(t.split())})
        for i, row in enumerate(e["table"]):
            text = header_to_text(row) if i == 0 else row_to_text(e["table"][0], row)
            units.append({"unit_id": f"{doc}::table_{i}", "doc_id": doc, "kind": "table", "text": text})
    return units


def build_questions(examples: list[dict], text_of: dict[str, str]) -> list[dict]:
    qs, mismatch = [], 0
    for e in examples:
        gold = [f"{e['filename']}::{k}" for k in e["qa"]["gold_inds"]]
        assert all(g in text_of for g in gold), e["id"]  # gold가 corpus에 반드시 있어야 한다
        for k, v in e["qa"]["gold_inds"].items():  # 우리 변환문 == FinQA gold 문장인지 (공백 무시)
            if k == "table_0":  # 헤더는 일부러 다르게 표현하므로 이 검사에서 제외 (header_to_text 참고)
                continue
            mismatch += text_of[f"{e['filename']}::{k}"].replace(" ", "") != v.replace(" ", "")
        qs.append({"qid": e["id"], "question": e["qa"]["question"], "doc_id": e["filename"],
                   "gold_unit_ids": gold, "program": e["qa"]["program"],
                   "answer": e["qa"]["answer"], "exe_ans": e["qa"]["exe_ans"]})
    assert mismatch == 0, f"변환문과 gold 문장이 다른 경우 {mismatch}건"
    return qs


def ambiguity(qs: list[dict], units: list[dict]) -> dict:
    """전체 합침(pooled) 검색에서 unit 텍스트만으로는 다른 회사 문서와 구분되지 않는 gold 비율.
    exact: 다른 filing에 완전히 같은 텍스트가 있음 / masked: 숫자만 다르고 나머지가 같음(템플릿 충돌)"""
    mask = lambda s: re.sub(r"\d", "#", s.replace(" ", ""))
    exact, masked = collections.defaultdict(set), collections.defaultdict(set)
    for u in units:
        exact[u["text"]].add(u["doc_id"])
        masked[mask(u["text"])].add(u["doc_id"])
    text_of = {u["unit_id"]: u["text"] for u in units}
    out = {}
    for name, idx, f in [("exact", exact, lambda s: s), ("masked", masked, mask)]:
        hit = sum(any(len(idx[f(text_of[g])]) > 1 for g in q["gold_unit_ids"]) for q in qs)
        out[f"{name}_dup_question_ratio"] = round(hit / len(qs), 3)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    raw_path, sha = download()
    examples = json.load(open(raw_path, encoding="utf-8"))
    units = build_corpus(examples)
    text_of = {u["unit_id"]: u["text"] for u in units}
    all_qs = build_questions(examples, text_of)
    subset = sorted(random.Random(SEED).sample(range(len(all_qs)), N_SUBSET))
    sub_qs = [all_qs[i] for i in subset]

    PROC.mkdir(parents=True, exist_ok=True)
    write_jsonl(PROC / "corpus.jsonl", units)
    write_jsonl(PROC / "questions_test_all.jsonl", all_qs)
    write_jsonl(PROC / f"questions_test{N_SUBSET}.jsonl", sub_qs)

    per_doc = collections.Counter(u["doc_id"] for u in units)
    n_gold = collections.Counter(len(q["gold_unit_ids"]) for q in sub_qs)
    manifest = {
        "source_url": URL, "sha256": sha, "seed": SEED,
        "n_docs": len(per_doc), "n_units": len(units),
        "units_per_doc": {"min": min(per_doc.values()), "median": statistics.median(per_doc.values()),
                          "max": max(per_doc.values())},
        "n_questions_all": len(all_qs), "n_questions_subset": len(sub_qs),
        "subset_gold_count_dist": dict(sorted(n_gold.items())),
        "ambiguity_all": ambiguity(all_qs, units),
        "ambiguity_subset": ambiguity(sub_qs, units),
    }
    (PROC / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    q = sub_qs[0]
    print("\n--- 예시 질문 ---\nQ:", q["question"], "\nprogram:", q["program"], "| exe_ans:", q["exe_ans"])
    for g in q["gold_unit_ids"]:
        print(" gold:", g, "->", text_of[g])


if __name__ == "__main__":
    main()
