from dotenv import load_dotenv
load_dotenv()

import os
import datetime
from duckduckgo_search import DDGS
import wikipedia
from langchain_openai import ChatOpenAI

wikipedia.set_lang("ko")

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 1. 데이터 수집 ──────────────────────────────────────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo 뉴스 + 일반 검색"""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            results.append({"source": "news", "title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("url", "")})
        for r in ddgs.text(query, max_results=max_results):
            results.append({"source": "web", "title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("url", "")})
    return results


def search_wikipedia(query: str) -> str:
    """Wikipedia 요약"""
    try:
        return wikipedia.summary(query, sentences=5)
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.summary(e.options[0], sentences=5)
        except Exception:
            return ""
    except Exception:
        return ""


def collect_data(topic: str) -> dict:
    print(f"\n  [1/3] 데이터 수집 중...")

    market_results = search_web(f"{topic} 시장 동향 2024 2025")
    competitor_results = search_web(f"{topic} 경쟁사 분석 주요 기업")
    wiki_summary = search_wikipedia(topic)

    print(f"    뉴스/웹: {len(market_results) + len(competitor_results)}건, Wikipedia: {'수집' if wiki_summary else '없음'}")
    return {
        "topic": topic,
        "market": market_results,
        "competitors": competitor_results,
        "wikipedia": wiki_summary,
    }


# ─── 2. 분석 및 보고서 생성 ──────────────────────────────────────

def format_search_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"- [{r['title']}]({r['url']})\n  {r['body'][:200]}")
    return "\n".join(lines)


def generate_report(data: dict) -> str:
    print(f"  [2/3] 보고서 생성 중...")

    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0.3)

    market_text = format_search_results(data["market"])
    competitor_text = format_search_results(data["competitors"])
    wiki_text = data["wikipedia"]

    prompt = f"""당신은 전문 시장 분석가입니다. 아래 수집된 정보를 바탕으로 '{data["topic"]}'에 대한 전문적인 분석 보고서를 작성하세요.

## 수집 데이터

### Wikipedia 개요
{wiki_text if wiki_text else "정보 없음"}

### 시장 동향 (뉴스/웹)
{market_text}

### 경쟁사 정보
{competitor_text}

---

## 보고서 작성 지침
아래 구조로 마크다운 보고서를 작성하세요. 수집된 데이터에 근거하여 작성하고, 전략적 인사이트를 포함하세요.

# {data["topic"]} 시장 동향 및 경쟁사 분석 보고서

## 1. 개요
(2~3문장으로 해당 시장/주제 소개)

## 2. 시장 동향
(최신 트렌드, 성장률, 주요 이슈 등 3~5개 항목)

## 3. 주요 경쟁사 분석
(주요 플레이어, 강점/약점, 포지셔닝 등)

## 4. 기회 및 위협 요인
(SWOT 관점에서 기회와 위협 각 2~3개)

## 5. 전략적 시사점
(실행 가능한 인사이트 3개 이상)

---
*보고서 생성: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    response = llm.invoke(prompt)
    return response.content


# ─── 3. 저장 및 출력 ─────────────────────────────────────────────

def save_report(topic: str, report: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = topic.replace(" ", "_").replace("/", "_")[:30]
    filename = f"{OUTPUT_DIR}/{safe_name}_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    return filename


# ─── 메인 ────────────────────────────────────────────────────────

def main():
    print("=== 시장 동향 및 경쟁사 분석 에이전트 ===\n")
    topic = input("분석 대상을 입력하세요 (예: 삼성전자 스마트폰, 국내 OTT 시장): ").strip()
    if not topic:
        print("입력이 없습니다.")
        return

    print(f"\n'{topic}' 분석을 시작합니다...")

    # 1. 수집
    data = collect_data(topic)

    # 2. 분석 + 보고서 생성
    report = generate_report(data)

    # 3. 저장
    print(f"  [3/3] 저장 중...")
    filepath = save_report(topic, report)

    # 출력
    print(f"\n{'='*60}")
    print(report)
    print(f"{'='*60}")
    print(f"\n보고서 저장됨: {filepath}\n")


if __name__ == "__main__":
    main()
