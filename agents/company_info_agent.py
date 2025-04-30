from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

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
        embedding=OpenAIEmbeddings(),
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

# 웹서치 도구를 통해 최신 뉴스 결과를 가져와 Document 형태로 반환
def get_web_news_fallback(query: str, k: int = 2) -> list[Document]:
    search = TavilySearchResults(k=k)
    results = search.invoke(query)
    return [
        Document(
            page_content=f"{item.get('content')}\n\n링크: {item.get('url')}",
            metadata={"type": "웹서치", "source": query, "agent_type": "startup_agent", "company_name": query.split()[0]}
        )
        
        
        for item in results
    ]


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

    # 2. 웹 뉴스
    web_news_docs = get_web_news_fallback(f"{startup_name} 스타트업 뉴스")

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
    # 테스트용 입력 상태 정의
    test_state = {
        "current_step": "스타트업_정보_수집",
        "startup_list": [{"name": "seedn"}],
        "selected_startup": {"name": "seedn"},
        "messages": [],
        "startup_info": None,
        "tech_info": None,
        "investment_decision": None,
        "report": None,
        "all_investment_decisions": {},
        "processed_startups_count": 0,
        "total_startups_count": 1,
    }

    # 함수 실행
    new_state = collect_startup_info(test_state)

    # 결과 출력
    from pprint import pprint
    pprint(new_state["startup_info"])
    print("메시지 로그:")
    for msg in new_state["messages"]:
        print(f"{msg['role']}: {msg['content']}")