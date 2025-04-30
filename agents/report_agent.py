# report_generation.py

import os
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import chromadb

# AgentState 타입 정의
AgentState = Dict[str, Any]

# 1) .env 파일에서 OpenAI API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("SUB_OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("환경변수 OPENAI_API_KEY 또는 SUB_OPENAI_API_KEY를 설정하세요.")
    print("주 API 키를 찾을 수 없어 대체 API 키를 사용합니다.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 2) Chat 모델 초기화
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

# 3) ChromaDB 초기화 및 컬렉션 접근
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("company_data")

# 4) 특정 회사의 모든 에이전트 정보 조회 함수
def get_company_data(company_name: str) -> Dict[str, Any]:
    """
    특정 회사명에 대한 모든 에이전트의 데이터를 조회합니다.
    """
    company_data = {}
    
    # startup_agent 데이터 조회
    startup_results = collection.query(
        query_texts=[company_name],
        where={
            "agent_type": "startup_agent",
            "company_name": company_name
        },
        n_results=5
    )
    
    # tech_agent 데이터 조회
    tech_results = collection.query(
        query_texts=[company_name],
        where={
            "agent_type": "tech_agent",
            "company_name": company_name
        },
        n_results=5
    )
    
    # invest_agent 데이터 조회
    invest_results = collection.query(
        query_texts=[company_name],
        where={
            "agent_type": "invest_agent",
            "company_name": company_name
        },
        n_results=5
    )
    
    # 조회된 데이터 병합 및 구조화
    if startup_results["documents"]:
        company_data["name"] = company_name
        company_data["news"] = "\n".join(startup_results["documents"])
        
        # 메타데이터에서 추가 정보 추출
        for metadata in startup_results["metadatas"]:
            if "revenue" in metadata:
                company_data["revenue"] = metadata.get("revenue", "")
            if "growth" in metadata:
                company_data["growth"] = metadata.get("growth", "")
            if "investment" in metadata:
                company_data["investment"] = metadata.get("investment", "")
            if "employees" in metadata:
                company_data["employees"] = metadata.get("employees", 0)
            if "patents" in metadata:
                company_data["patents"] = metadata.get("patents", 0)
    
    if tech_results["documents"]:
        company_data["tech_summary"] = "\n".join(tech_results["documents"])
    
    if invest_results["documents"]:
        company_data["reasoning"] = invest_results["documents"][0]
        # 투자 판단 메타데이터 추출
        if invest_results["metadatas"]:
            company_data["decision"] = invest_results["metadatas"][0].get("decision", "투자보류")
    
    return company_data


# 5) 투자추천 보고서용 프롬프트 템플릿
investment_template = """
아래는 '투자추천'으로 판단된 스타트업의 상세 정보입니다.

기업명: {name}

[재무 및 성장]
- 매출(억원): {revenue}
- 성장률(%): {growth}

[투자 유치]
- 라운드별 투자금액(억원): {investment}

[인력·특허]
- 고용인원: {employees}명
- 특허 건수: {patents}건

[주요 뉴스]
{news}

[핵심 기술 요약]
{tech_summary}

[투자 판단 및 근거]
{reasoning}

위 정보를 바탕으로, 투자 담당자가 바로 활용할 수 있는 '투자 보고서'를 작성하세요.
보고서에는 다음 항목이 포함되어야 합니다:
1. 기업 개요 및 핵심 경쟁력
2. 최근 성장성과 재무 지표 분석
3. 투자 유치 이력 및 시장 검증 수준
4. 기술력 및 지적재산권 현황
5. 투자 추천 근거 및 기대 효과
"""

investment_prompt = PromptTemplate(
    input_variables=[
        "name","revenue","growth","investment",
        "employees","patents","news","tech_summary","reasoning"
    ],
    template=investment_template
)
investment_chain = LLMChain(llm=llm, prompt=investment_prompt)


# 6) 전원 투자보류 종합 보고서용 프롬프트 템플릿
hold_template = """
아래는 심사한 스타트업 리스트와 각 기업의 투자보류 사유입니다.

{company_hold_reasons}

위 데이터를 바탕으로 '종합 투자보류 보고서'를 작성하세요.
보고서에는 다음을 포함해야 합니다:
1. 검토 기업 리스트 및 기본 정보 요약
2. 기업별 투자보류 핵심 사유 (수익성, 성장성, 시장 검증 등)
3. 투자보류 기업들의 공통적 한계점 도출
4. 투자심사 프로세스 및 기준 개선 제안
5. 향후 투자 전략 방향성 및 재검토 조건
"""

hold_prompt = PromptTemplate(
    input_variables=["company_hold_reasons"],
    template=hold_template
)
hold_chain = LLMChain(llm=llm, prompt=hold_prompt)


# 7) 보고서 생성 에이전트 정의
def report_generation_agent(state: AgentState) -> AgentState:
    """
    ChromaDB에서 특정 회사명으로 필터링한 데이터를 기반으로
    투자 보고서를 생성합니다.
    """
    company_name = state["selected_startup"]["name"]
    
    # ChromaDB에서 회사 데이터 가져오기
    company_data = get_company_data(company_name)
    
    results = []

    if not company_data:
        report = f"{company_name}에 대한 데이터가 없습니다."
    else:
        # 투자 결정에 따른 보고서 생성
        if company_data.get("decision") == "투자추천":
            report = investment_chain.run(
                name=company_data.get("name", ""),
                revenue=company_data.get("revenue", ""),
                growth=company_data.get("growth", ""),
                investment=company_data.get("investment", ""),
                employees=str(company_data.get("employees", "")),
                patents=str(company_data.get("patents", "")),
                news=company_data.get("news", ""),
                tech_summary=company_data.get("tech_summary", ""),
                reasoning=company_data.get("reasoning", "")
            )
        else:
            # 투자보류 시 해당 회사에 대한 보류 보고서 생성
            company_hold_reasons = f"- {company_data['name']}: {company_data.get('reasoning','')}"
            report = hold_chain.run(company_hold_reasons=company_hold_reasons)
            results.append({"name": f"{company_name} 투자보류 보고서", "report": report})
    
    return {
        **state,
        "report": report,
        "current_step": "보고서_생성",
        "messages": state["messages"] + [{"role": "system", "content": f"{company_name} 보고서 생성 완료"}]
    }

# 8) 여러 회사 데이터를 처리하여 종합 보고서 생성
def generate_summary_report(
    company_names: List[str]
) -> List[Dict[str, str]]:
    """
    여러 회사 데이터를 처리하여 개별 보고서 또는 종합 보고서를 생성합니다.
    """
    all_company_data = []
    invest_companies = []
    
    # 각 회사별 데이터 수집
    for company_name in company_names:
        data = get_company_data(company_name)
        if data:
            all_company_data.append(data)
            if data.get("decision") == "투자추천":
                invest_companies.append(data)
    
    results = []
    
    # 투자추천 기업이 있는 경우 개별 보고서 생성
    if invest_companies:
        for company in invest_companies:
            report = investment_chain.run(
                name=company.get("name", ""),
                revenue=company.get("revenue", ""),
                growth=company.get("growth", ""),
                investment=company.get("investment", ""),
                employees=str(company.get("employees", "")),
                patents=str(company.get("patents", "")),
                news=company.get("news", ""),
                tech_summary=company.get("tech_summary", ""),
                reasoning=company.get("reasoning", "")
            )
            results.append({"name": company["name"], "report": report})
    else:
        # 전원 투자보류 시 종합 보고서 생성
        lines = []
        for company in all_company_data:
            lines.append(f"- {company['name']}: {company.get('reasoning','')}")
        company_hold_reasons = "\n".join(lines)
        report = hold_chain.run(company_hold_reasons=company_hold_reasons)
        results.append({"name": "종합 투자보류 보고서", "report": report})
    
    return results


# 9) 실행 예시
if __name__ == "__main__":
    # 단일 회사 보고서 생성
    print("\n==== 단일 회사 보고서 ====")
    company_name = "카카오"
    reports = report_generation_agent(company_name)
    for item in reports:
        print(f"\n==== [{item['name']}] ====\n{item['report']}\n")
    
    # 여러 회사 종합 보고서 생성
    print("\n==== 종합 보고서 ====")
    company_names = ["카카오", "네이버", "삼성"]
    summary_reports = generate_summary_report(company_names)
    for item in summary_reports:
        print(f"\n==== [{item['name']}] ====\n{item['report']}\n")
