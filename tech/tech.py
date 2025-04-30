from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
import os
import re
import uuid
from typing import TypedDict, List, Dict, Optional, Literal

# 상태 타입 정의
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

# 🔍 기술 요약 파이프라인
def tech_exploration(state: AgentState, save_txt: bool = True) -> AgentState:
    company = state["selected_startup"]["name"]
    search = TavilySearchResults(k=5)
    results = search.run(f"{company} 기술력 OR 핵심 기술 OR AI OR IoT OR 특허")
    urls = [r["url"] for r in results]

    loader = WebBaseLoader(urls)
    docs = loader.load()
    texts = [doc.page_content for doc in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        for chunk in splitter.split_text(text):
            chunks.append(Document(page_content=chunk, metadata={
                "agent_type": "tech_chunk",
                "company_name": company
            }))

    # ✅ 벡터 저장소 생성
    persist_dir = "chroma_store"
    collection_name = "startup_tech_chunks"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)

    # ✅ AI 요약 수행
    llm = ChatOpenAI(temperature=0.2)
    summaries = []
    for text in texts:
        msg = llm.invoke(f"{text}\n\n이 기업의 기술적 경쟁력을 한 문장으로 요약해줘.")
        summaries.append(msg.content)

    # ✅ 요약도 벡터 DB에 저장
    summary_docs = [
        Document(page_content=s, metadata={
            "agent_type": "tech_summary",
            "company_name": company
        }) for s in summaries
    ]
    summary_collection_name = f"{uuid.uuid4().hex[:8]}_tech_summary"
    summary_vectorstore = Chroma.from_documents(
        documents=summary_docs,
        embedding=embeddings,
        collection_name=summary_collection_name,
        persist_directory=persist_dir
    )

    # ✅ 최종 요약도 벡터 DB에 저장
    final_response = llm.invoke("".join(summaries) + "\n\n이 스타트업의 기술적 경쟁력을 한 문장으로 요약해줘.")
    final_summary = final_response.content
    final_doc = Document(
        page_content=final_summary,
        metadata={"agent_type": "final_summary", "company_name": company}
    )
    summary_vectorstore.add_documents([final_doc])

    # ✅ TXT 백업 저장 (옵션)
    if save_txt:
        os.makedirs("text_backup", exist_ok=True)
        vector_txt_path = f"text_backup/{company}_vector_content.txt"
        all_docs = vectorstore.similarity_search("기술", k=100) + summary_vectorstore.similarity_search("기술", k=100)
        with open(vector_txt_path, "w", encoding="utf-8") as f:
            for i, d in enumerate(all_docs, 1):
                f.write(f"[{i}] {d.metadata.get('agent_type')} | {d.metadata.get('company_name')}\n{d.page_content}\n\n")
        print(f"✅ 벡터 DB 저장 내용 백업 완료: {vector_txt_path}")

    # ✅ 최종 요약 결과 저장
    state["tech_info"] = {
        "summary": final_summary,
        "source_urls": urls
    }
    return state

# 🧪 테스트 실행
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    company_name = "씨드앤"
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
