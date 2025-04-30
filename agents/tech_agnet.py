from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from bs4 import BeautifulSoup
import os
import uuid
from typing import TypedDict, List, Dict, Optional, Literal

class AgentState(TypedDict):
    current_step: str
    startup_list: List[Dict]
    selected_startup: Optional[Dict]
    startup_info: Optional[Dict]
    tech_info: Optional[Dict]
    investment_decision: Optional[Literal["투자추천", "투자보류"]]
    report: Optional[str]
    all_investment_decisions: Dict[str, Literal["투자추천", "투자보류"]]
    processed_startups_count: int
    total_startups_count: int
    messages: List[Dict]

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # ✅ 불필요한 패턴 필터링
    filters = [
        "무단 전재 및 재배포 금지",
        "저작권자",
        "관련기사",
        "SNS 공유하기",
        "광고 문의",
        "네이버 홈",
        "All rights reserved", "무단복제", 
        "기자의 다른 기사 보기", "기사제보", "페이스북", "트위터", "카카오스토리", 
        "네이버 뉴스", "본문 시작", "기사 본문", "닫기"
    ]
    for pattern in filters:
        text = text.replace(pattern, "")
    
    # ✅ 공백 정리
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def tech_exploration(state: AgentState, save_txt: bool = True) -> AgentState:
    company = state["selected_startup"]["name"]
    search = TavilySearchResults(k=5)
    results = search.run(f"{company}의 핵심 기술, AI 또는 IoT 관련 기술력, 그리고 특허 기술에 대해 알고싶어")
    urls = [r["url"] for r in results]

    loader = WebBaseLoader(urls)
    docs = loader.load()

    # ✅ HTML 정제 적용
    texts = [clean_html_text(doc.page_content) for doc in docs]

    llm = ChatOpenAI(temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    summaries = []
    for text in texts:
        msg = llm.invoke(f"{text}\n\n이 기업의 기술을 요약해줘.")
        summaries.append(msg.content)

    summary_collection_name = f"{uuid.uuid4().hex[:8]}_tech_summary"
    persist_dir = "../chroma_store"
    summary_docs = [
        Document(page_content=s, metadata={
            "agent_type": "tech_summary",
            "company_name": company
        }) for s in summaries
    ]
    summary_vectorstore = Chroma.from_documents(
        documents=summary_docs,
        embedding=embeddings,
        collection_name=summary_collection_name,
        persist_directory=persist_dir
    )

    final_response = llm.invoke("".join(summaries) + "\n\n이 스타트업의 기술적 경쟁력은 무엇인가요?")
    final_summary = final_response.content
    final_doc = Document(
        page_content=final_summary,
        metadata={"agent_type": "tech_agent", "company_name": company}
    )
    summary_vectorstore.add_documents([final_doc])

    if save_txt:
        os.makedirs("../text_backup", exist_ok=True)
        vector_txt_path = f"../text_backup/{company}_vector_content.txt"
        all_docs = summary_vectorstore.similarity_search("기술", k=100)
        with open(vector_txt_path, "w", encoding="utf-8") as f:
            for i, d in enumerate(all_docs, 1):
                f.write(f"[{i}] {d.metadata.get('agent_type')} | {d.metadata.get('company_name')}\n{d.page_content}\n\n")
        print(f"✅ 벡터 DB 저장 내용 백업 완료: {vector_txt_path}")

    state["tech_info"] = {
        "summary": final_summary,
        "source_urls": urls
    }
    return state

# 🧪 테스트 실행
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    company_name = "프리윌린"
    test_state = {
        "current_step": "기술_탐색_요약",
        "startup_list": [],
        "selected_startup": {"name": company_name},
        "startup_info": None,
        "tech_info": None,
        "investment_decision": None,
        "report": None,
        "all_investment_decisions": {},
        "processed_startups_count": 0,
        "total_startups_count": 10,
        "messages": []
    }

    result_state = tech_exploration(test_state, save_txt=True)

    print("\n📌 최종 기술 요약 응답:", result_state["tech_info"]["summary"])
    print("\n🔗 수집된 기사 URL:")
    for url in result_state["tech_info"]["source_urls"]:
        print(" -", url)