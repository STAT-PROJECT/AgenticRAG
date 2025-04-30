"""
투자 판단 에이전트 - 스타트업의 재무, 기술, 성장성 등을 종합적으로 분석하여 투자 결정을 내림
"""
import os
from typing import Dict, Any, List, Tuple, Literal
from dotenv import load_dotenv
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

# .env 파일에서 OpenAI API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("SUB_OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("환경변수 OPENAI_API_KEY 또는 SUB_OPENAI_API_KEY를 설정하세요.")
    print("주 API 키를 찾을 수 없어 대체 API 키를 사용합니다.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# LLM 모델 초기화
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# ChromaDB 초기화
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("company_data")


def get_startup_info(company_name: str) -> Dict[str, Any]:
    """
    스타트업 정보 수집 에이전트가 수집한 데이터를 ChromaDB에서 조회
    """
    results = collection.query(
        query_texts=[company_name],
        where={
            "agent_type": "startup_agent",
            "company_name": company_name
        },
        n_results=10
    )
    
    # 스타트업 기본 정보 구조화
    startup_info = {
        "name": company_name,
        "news": [],
        "revenue": None,
        "growth": None,
        "investment": None,
        "employees": None,
        "patents": None,
        "assets": None,
        "capital": None,
        "operating_profit": None,
        "net_profit": None
    }
    
    # 문서와 메타데이터 처리
    if results["documents"] and len(results["documents"]) > 0:
        startup_info["news"] = results["documents"]
        
        # 메타데이터에서 추가 정보 추출
        for metadata in results["metadatas"]:
            for key in startup_info.keys():
                if key in metadata and metadata[key] is not None:
                    startup_info[key] = metadata[key]
    
    return startup_info


def get_tech_info(company_name: str) -> Dict[str, Any]:
    """
    기술 탐색 요약 에이전트가 분석한 기술 정보를 ChromaDB에서 조회
    """
    results = collection.query(
        query_texts=[company_name],
        where={
            "agent_type": "tech_agent",
            "company_name": company_name
        },
        n_results=5
    )
    
    tech_info = {
        "tech_summary": "",
        "core_technology": "",
        "market_position": "",
        "competitors": "",
        "tech_advantages": ""
    }
    
    if results["documents"] and len(results["documents"]) > 0:
        tech_info["tech_summary"] = "\n".join(results["documents"])
        
        # 메타데이터에서 추가 기술 정보 추출
        for metadata in results["metadatas"]:
            for key in tech_info.keys():
                if key in metadata and metadata[key] is not None:
                    tech_info[key] = metadata[key]
    
    return tech_info


# 투자 판단을 위한 프롬프트 템플릿
investment_analysis_template = """
스타트업 투자 분석 전문가로서, 아래 스타트업에 대한 시리즈 B 투자 적합성을 분석해 주세요.

# 기업 정보
- 기업명: {name}

# 재무 및 성장성 정보
- 매출: {revenue}
- 성장률: {growth}
- 영업이익: {operating_profit}
- 순이익: {net_profit}
- 자산: {assets}
- 자본: {capital}

# 투자 유치 이력
- 투자유치 내역: {investment}

# 인력 및 특허
- 고용인원: {employees}
- 특허 건수: {patents}

# 뉴스 및 언론 보도
{news_summary}

# 기술 정보
{tech_summary}

# 시리즈 B 투자 판단 기준
1. 성장성: 매출 성장률이 연간 최소 50% 이상이며, 지속적 성장 트렌드 유지
2. 수익성: 매출 대비 손실 비율이 감소 추세이며, 손익분기점 계획이 명확함
3. 시장성: 글로벌 확장 또는 대규모 시장 진입 가능성
4. 기술력: 핵심 기술의 차별화 및 지적재산권 보유
5. 투자효율성: 기존 투자금 대비 성과(매출, 성장률) 증명
6. 팀 역량: 50인 이상 조직 운영 능력, 핵심 인재 확보
7. 비즈니스 지표: 고객 획득비용(CAC), 고객생애가치(LTV), 재구매율 등 긍정적

위 정보를 종합적으로 분석하여 다음 양식으로 투자 판단 결과를 작성해 주세요:

## 실질적인 성과(Traction) 평가
[성장성, 매출 추이, 검증된 사업 성과 등 분석]

## 수익성과 단위 경제학 분석
[재무 상태 변화, 손익 추이, 수익모델 분석]

## 확장 전략 관련 정보
[투자 유치 이력, 자금 활용 계획, 확장성]

## 비즈니스 지표 관리
[핵심 지표, 조직 운영 현황]

## 지속 가능한 성장의 근거
[기술력, 시장 포지셔닝, 차별화 요소]

## 종합 분석 및 투자 판단
[전체 분석을 바탕으로 한 최종 판단]

## 투자 판단 수치적 근거
[투자 판단에 대한 구체적 수치와 근거]

최종 판단: "투자추천" 또는 "투자보류" 중 하나로만 답변하세요.
"""

investment_analysis_prompt = PromptTemplate(
    input_variables=[
        "name", "revenue", "growth", "operating_profit", "net_profit",
        "assets", "capital", "investment", "employees", "patents",
        "news_summary", "tech_summary"
    ],
    template=investment_analysis_template
)

investment_analysis_chain = LLMChain(llm=llm, prompt=investment_analysis_prompt)


def analyze_investment(startup_info: Dict[str, Any], tech_info: Dict[str, Any]) -> Tuple[str, str]:
    """
    스타트업 정보와 기술 정보를 바탕으로 투자 분석을 수행하고 결정을 내림
    반환: (결정, 분석 결과)
    """
    # 뉴스 요약 생성
    news_summary = "\n".join(startup_info["news"][:3]) if startup_info["news"] else "관련 뉴스 정보가 없습니다."
    
    # 분석 실행
    analysis_result = investment_analysis_chain.run(
        name=startup_info["name"],
        revenue=startup_info.get("revenue", "정보 없음"),
        growth=startup_info.get("growth", "정보 없음"),
        operating_profit=startup_info.get("operating_profit", "정보 없음"),
        net_profit=startup_info.get("net_profit", "정보 없음"),
        assets=startup_info.get("assets", "정보 없음"),
        capital=startup_info.get("capital", "정보 없음"),
        investment=startup_info.get("investment", "정보 없음"),
        employees=startup_info.get("employees", "정보 없음"),
        patents=startup_info.get("patents", "정보 없음"),
        news_summary=news_summary,
        tech_summary=tech_info.get("tech_summary", "기술 정보가 없습니다.")
    )
    
    # 결과에서 최종 판단 추출
    decision = "투자보류"  # 기본값
    
    # 분석 결과의 마지막 부분에서 투자 판단 추출
    analysis_lines = analysis_result.strip().split("\n")
    for line in reversed(analysis_lines):
        if "투자추천" in line:
            decision = "투자추천"
            break
        elif "투자보류" in line:
            decision = "투자보류"
            break
    
    return decision, analysis_result


def save_investment_decision(company_name: str, decision: str, analysis: str) -> None:
    """
    투자 판단 결과를 ChromaDB에 저장
    """
    # 분석 결과 내용 중 일부를 메타데이터로 추출
    analysis_sections = analysis.split("##")
    summary = analysis_sections[-2] if len(analysis_sections) > 2 else "종합 분석 없음"
    
    # ChromaDB에 저장
    collection.add(
        documents=[analysis],
        metadatas=[{
            "agent_type": "invest_agent",
            "company_name": company_name,
            "decision": decision,
            "summary": summary[:500] if len(summary) > 500 else summary  # 메타데이터 길이 제한
        }],
        ids=[f"invest_analysis_{company_name}_{decision}"]
    )


def investment_decision_agent(company_name: str) -> Dict[str, Any]:
    """
    투자 판단 에이전트의 메인 함수
    """
    print(f"[투자 판단 에이전트] {company_name} 기업에 대한 투자 판단 시작")
    
    # 스타트업 정보와 기술 정보 조회
    startup_info = get_startup_info(company_name)
    tech_info = get_tech_info(company_name)
    
    # 정보가 부족한 경우 처리
    if not startup_info.get("revenue") or not tech_info.get("tech_summary"):
        print(f"[투자 판단 에이전트] {company_name} 기업 정보 부족으로 투자보류 판단")
        decision = "투자보류"
        analysis = f"{company_name} 기업에 대한 충분한 정보가 없어 투자 판단을 할 수 없습니다. 추가 정보 수집이 필요합니다."
    else:
        # 투자 분석 수행
        decision, analysis = analyze_investment(startup_info, tech_info)
    
    # 결과 저장
    save_investment_decision(company_name, decision, analysis)
    
    print(f"[투자 판단 에이전트] {company_name} 기업 투자 판단 결과: {decision}")
    
    # 결과 반환
    return {
        "company_name": company_name,
        "decision": decision,
        "analysis": analysis
    }


# 단독 실행용 코드
if __name__ == "__main__":
    # 테스트용 코드
    test_company = "시드앤"
    result = investment_decision_agent(test_company)
    print(f"투자 결정: {result['decision']}")
    print("분석 요약:")
    print(result['analysis'][:500] + "...")  # 분석 결과 일부만 출력