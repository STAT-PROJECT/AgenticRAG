from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup 

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

load_dotenv()


# PDF 파일을 텍스트로 로드하는 함수
def load_pdf_text(file_path: str) -> str:                   
    loader = PyMuPDFLoader(file_path)                       # PDF 로더 초기화
    pages = loader.load()                                   # 모든 페이지 로드
    return "\n".join([p.page_content for p in pages])       # 페이지 내용을 하나의 문자열로 결합
    
# LLM을 사용해 구조화된 정보를 추출하는 함수
def extract_structured_info(text: str) -> str:
    with open("prompts/startup_info_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)         # LLM 모델 지정
    prompt = PromptTemplate.from_template(template)         # 템플릿을 Prompt 객체로 변환
    chain = prompt | llm | StrOutputParser()                # 프롬프트 → LLM → 출력 파서 체인 구성
    return chain.invoke({"text": text})                     # LLM에 텍스트 전달하고 결과 반환

# 구조화된 텍스트를 Document 리스트로 변환 (Chroma 저장용)
def split_to_documents(extracted_text: str, pdf_name: str, company_name: str) -> list[Document]:
    sections = extracted_text.split("\n---")[0].split("\n") # "---" 구분 전까지 섹션 추출
    documents = []
    current_section = ""
    buffer = []

    for line in sections:
        if not line.strip():
            continue
        if ":" in line and not line.startswith("  "):       # 새로운 섹션의 시작
            if buffer:
                documents.append(
                    Document(
                        page_content="\n".join(buffer),
                        metadata={"type": current_section, "source": pdf_name, "agent_type": "startup_agent", "company_name": company_name}
                    )
                )
            current_section, value = line.split(":", 1)
            buffer = [f"{current_section.strip()}: {value.strip()}"]
        else:
            buffer.append(line.strip())
    # 마지막 버퍼 내용 저장
    if buffer:
        documents.append(
            Document(
                page_content="\n".join(buffer),
                metadata={"type": current_section, "source": pdf_name, "agent_type": "startup_agent", "company_name": company_name}
            )
        )
    return documents

# ChromaDB에 Document 저장 (collection_name에 저장)
def store_in_chroma(documents: list[Document], collection_name: str, persist_dir: str = "chroma_store"):
    os.makedirs(persist_dir, exist_ok=True)
    
    # 문서별 고유 ID 생성 (회사명, 문서타입, 인덱스를 조합)
    ids = []
    for i, doc in enumerate(documents):
        company = doc.metadata.get("company_name") or "unknown"
        doc_type = doc.metadata.get("type") or f"doc{i}"
        ids.append(f"{company}_{doc_type}_{i}")

    # 벡터 저장소에 문서 저장
    chroma = Chroma.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_dir,
        collection_name=collection_name,
        ids=ids
    )
    chroma.persist()

# 구조화된 텍스트 + 뉴스 결과를 TXT 파일로 저장
def save_structured_text_to_file(startup_name: str, structured_text: str, web_news_docs: list[Document] = None, output_dir="data/outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{startup_name}_info.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(structured_text)
        f.write("\n\n" + "="*50 + "\n")
        f.write("웹서치 뉴스 요약 결과\n")
        f.write("="*50 + "\n\n")

        if web_news_docs:
            for i, doc in enumerate(web_news_docs, 1):
                f.write(f"{i}. {doc.page_content.strip()}\n\n")
        else:
            f.write("웹서치 뉴스 없음\n")

    return output_path

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # 불필요한 패턴 필터링
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
    
    # 공백 정리
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def get_web_news_docs(company: str, save_txt: bool = True) -> list[Document]:
    search = TavilySearchResults(k=5)
    results = search.run(f"{company} 관련 뉴스 OR 보도자료")
    urls = [r["url"] for r in results]

    loader = WebBaseLoader(urls)
    docs = loader.load()

    texts = [clean_html_text(doc.page_content) for doc in docs]

    llm = ChatOpenAI(temperature=0.2)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    summaries = []
    for text in texts:
        msg = llm.invoke(f"{text}\n\n이 뉴스 내용을 핵심 내용이 누락되지 않게 요약해줘.")
        summaries.append(msg.content)

    news_docs = [
        Document(page_content=s, metadata={
            "agent_type": "news_summary",
            "company_name": company,
            "source": urls[i] if i < len(urls) else "unknown"
        }) for i, s in enumerate(summaries)
    ]

    if save_txt:
        os.makedirs("data/outputs", exist_ok=True)
        vector_txt_path = f"data/outputs/{company}_news_vector_content.txt"
        with open(vector_txt_path, "w", encoding="utf-8") as f:
            for i, d in enumerate(news_docs, 1):
                f.write(f"[{i}] {d.metadata.get('agent_type')} | {d.metadata.get('company_name')}\n{d.page_content}\n\n")
        print(f"뉴스 백업 저장 완료: {vector_txt_path}")

    return news_docs


def collect_startup_info(state: AgentState) -> AgentState:
    startup = state["selected_startup"]
    if not startup:
        raise ValueError("선택된 스타트업 정보가 없습니다.")

    startup_name = startup["name"]
    pdf_path = f"data/{startup_name}.pdf"

    if not os.path.exists(pdf_path):
        return {
            **state,
            "startup_info": {"status": "0", "reason": f"{pdf_path} 없음"},
            "messages": state["messages"] + [{"role": "system", "content": f"{startup_name} PDF 없음"}],
        }

    # 1. PDF + 요약
    raw_text = load_pdf_text(pdf_path)
    structured_text = extract_structured_info(raw_text)

    # 2. 뉴스 수집 및 요약
    web_news_docs = get_web_news_docs(startup_name)  # 수정된 함수로 실행


    # 3. 저장
    text_path = save_structured_text_to_file(startup_name, structured_text, web_news_docs)
    documents = split_to_documents(structured_text, pdf_name=pdf_path, company_name=startup_name)
    store_in_chroma(documents, collection_name="company_data")
    store_in_chroma(web_news_docs, collection_name="company_data")

    return {
        **state,
        "startup_info": {
            "status": "1",
            "company": startup_name,
            "pdf": pdf_path,
            "text_path": text_path
        },
        "messages": state["messages"] + [{"role": "system", "content": f"{startup_name} 정보 수집 완료"}],
    }

if __name__ == "__main__":
    from pprint import pprint

    # 테스트 기업명 (한글 이름 기준)
    company_name = "씨드앤"

    # 테스트용 상태 정의
    test_state = {
        "current_step": "스타트업_정보_수집",
        "startup_list": [{"name": company_name}],
        "selected_startup": {"name": company_name},
        "messages": [],
        "startup_info": None,
        "tech_info": None,
        "investment_decision": None,
        "report": None,
        "all_investment_decisions": {},
        "processed_startups_count": 0,
        "total_startups_count": 1,
    }

    # 실행
    updated_state = collect_startup_info(test_state)

    # 출력
    print("\n수집 완료된 startup_info:")
    pprint(updated_state["startup_info"])

    print("\n메시지 로그:")
    for msg in updated_state["messages"]:
        print(f"{msg['role']}: {msg['content']}")