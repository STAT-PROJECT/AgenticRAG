import os
from typing import Dict, Any, List, Optional
import json
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
import requests
from chromadb.utils import embedding_functions
import time

# .env 파일에서 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = os.getenv("SUB_OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("환경변수 OPENAI_API_KEY 또는 SUB_OPENAI_API_KEY를 설정하세요.")
    print("주 API 키를 찾을 수 없어 대체 API 키를 사용합니다.")

# Tavily API 키 설정
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    print("경고: TAVILY_API_KEY가 설정되지 않았습니다. 일부 기능이 작동하지 않을 수 있습니다.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 한국어 임베딩을 위한 대체 모델 사용
# jhgan/ko-sroberta-multitask는 한국어 문장 임베딩에 적합한 모델입니다
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

# ChromaDB 초기화 (임베딩 함수 지정)
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(
    name="company_data",
    embedding_function=hf_ef
)


def get_tavily_search_results(query: str) -> Dict:
    """
    Tavily API를 직접 호출하여 검색 결과를 가져옵니다.
    """
    if not TAVILY_API_KEY:
        return {"error": "Tavily API 키가 설정되지 않았습니다."}
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": 5,
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Tavily API 호출 중 오류 발생: {e}")
        return {"error": str(e)}


def get_tavily_market_data(company_name: str) -> str:
    """
    Tavily API를 사용하여 시장 포지셔닝 및 경쟁사 정보를 검색합니다.
    """
    if not TAVILY_API_KEY:
        return "Tavily API 키가 설정되지 않아 검색을 수행할 수 없습니다."
    
    try:
        # 다양한 검색 쿼리 시도
        queries = [
            f"{company_name} 시장 포지셔닝",
            f"{company_name} 경쟁사 분석",
            f"{company_name} 차별화 전략",
            f"{company_name} 주요 경쟁사",
            f"{company_name} 시장 점유율",
            f"{company_name} 산업 분석"
        ]
        
        results = []
        for query in queries:
            search_results = get_tavily_search_results(query)
            if "error" in search_results:
                results.append(f"쿼리: {query}\n검색 오류: {search_results['error']}")
            else:
                # 응답 형식 가공
                results.append(f"쿼리: {query}\n검색 결과: {json.dumps(search_results, ensure_ascii=False, indent=2)}")
            print(f"Tavily 검색 완료: {query}")
            time.sleep(1)  # API 요청 간 짧은 대기 시간
        
        # 결과 통합
        combined_results = "\n\n" + "\n\n".join(results)
        return combined_results
    except Exception as e:
        print(f"Tavily 검색 중 오류 발생: {e}")
        return f"검색 중 오류 발생: {str(e)}"


def generate_market_analysis(company_name: str, tavily_results: str) -> str:
    """
    OpenAI API를 직접 사용하여 시장 분석을 생성합니다.
    정보가 충분하지 않을 경우 명확히 표시합니다.
    """
    # 검색 결과 품질 확인
    has_sufficient_data = check_search_quality(tavily_results)
    
    prompt = f"""
다음은 {company_name} 회사에 대해 수집된 정보입니다.

[검색 결과]
{tavily_results}

위 정보를 바탕으로 {company_name}의 시장 포지셔닝 및 경쟁사 분석을 수행해주세요.
결과는 다음 구조로 작성해주세요:

## 시장 포지셔닝
[회사가 속한 시장/산업과 그 시장에서의 위치를 설명]

## 주요 경쟁사
[주요 경쟁사 3-5개를 나열하고 각각에 대해 간략히 설명]

## 경쟁 우위 요소
[해당 기업이 경쟁사 대비 가지는 차별화 요소나 강점]

## 시장 점유율 및 성장성
[가능한 경우 시장 점유율 정보와 향후 성장 가능성]

## 위협 요소
[시장에서 직면한 주요 도전 과제나 위협 요인]

중요 지침:
1. 답변은 반드시 검색 결과에서 확인된 정보만 사용하세요.
2. 특정 정보가 검색 결과에 명확히 나타나지 않으면 반드시 "정보 없음" 또는 "검색 결과에서 확인할 수 없음"이라고 명시하세요.
3. 추측하거나 일반적인 내용으로 빈 칸을 채우지 마세요.
4. 각 섹션에서 최소 하나 이상의 구체적인 출처(예: '00 기사에 따르면')를 포함하세요.

불확실하거나 정보가 부족한 경우에는 반드시 그 사실을 명시해야 합니다.
"""

    try:
        # 데이터 충분성에 따른 시스템 메시지 조정
        system_message = "당신은 시장 분석 전문가로서 정확하고 객관적인 분석을 제공합니다."
        if not has_sufficient_data:
            system_message += " 정보가 불충분한 경우 추측을 피하고 반드시 '정보 없음'을 명시해야 합니다."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # 더 결정적인 응답을 위해 온도 낮춤
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return f"분석 중 오류 발생: {str(e)}"

def check_search_quality(search_results: str) -> bool:
    """
    검색 결과의 품질을 평가하여 충분한 정보가 있는지 확인합니다.
    """
    # 검색 응답 길이 확인
    if len(search_results) < 1000:  # 임의의 기준
        return False
    
    # 오류 메시지 확인
    if "검색 오류" in search_results and search_results.count("검색 오류") > 3:
        return False
        
    # 실질적인 정보 포함 여부 확인 (JSON 응답에 의미있는 내용이 있는지)
    meaningful_data_markers = [
        '"answer":', 
        '"content":', 
        '"snippet":'
    ]
    
    meaningful_data_count = sum(1 for marker in meaningful_data_markers if marker in search_results)
    if meaningful_data_count < 2:  # 최소 두 개 이상의 의미 있는 데이터 마커가 있어야 함
        return False
    
    return True

def analyze_market_position(company_name: str) -> Dict[str, Any]:
    """
    시장 포지셔닝 및 경쟁사 분석을 종합적으로 수행합니다.
    """
    print(f"\n[시장 분석 에이전트] {company_name} 기업의 시장 포지셔닝 및 경쟁사 분석 시작")
    
    # Tavily 검색 결과
    tavily_results = get_tavily_market_data(company_name)
    
    # 검색 결과 품질 평가
    has_sufficient_data = check_search_quality(tavily_results)
    if not has_sufficient_data:
        print(f"[시장 분석 에이전트] ⚠️ {company_name} 기업에 대한 검색 결과가 충분하지 않습니다. 제한된 분석이 수행됩니다.")
    
    # OpenAI를 사용한 종합 분석
    market_analysis = generate_market_analysis(company_name, tavily_results)
    
    print(f"[시장 분석 에이전트] {company_name} 기업 시장 분석 완료")
    
    # 결과 저장
    save_market_analysis(company_name, market_analysis, has_sufficient_data)
    
    return {
        "company_name": company_name,
        "market_analysis": market_analysis,
        "has_sufficient_data": has_sufficient_data
    }


def save_market_analysis(company_name: str, analysis: str, has_sufficient_data: bool = True) -> None:
    """
    시장 분석 결과를 섹션별로 청크로 나누어 ChromaDB에 저장합니다.
    """
    try:
        # 분석 결과를 섹션으로 분할
        sections = analysis.split("##")
        
        # 첫 번째 빈 섹션 제거
        if sections and not sections[0].strip():
            sections = sections[1:]
            
        documents = []
        metadatas = []
        ids = []
        
        # 전체 분석 결과도 하나의 문서로 저장 (통합 검색용)
        documents.append(analysis)
        metadatas.append({
            "agent_type": "market_agent",
            "company_name": company_name,
            "content_type": "full_analysis",
            "has_sufficient_data": str(has_sufficient_data)
        })
        ids.append(f"market_analysis_{company_name}_full")
        
        # 각 섹션을 별도의 청크로 저장
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # 섹션 제목과 내용 분리
            lines = section.strip().split("\n", 1)
            if len(lines) < 2:
                continue
                
            section_title = lines[0].strip()
            section_content = lines[1].strip() if len(lines) > 1 else ""
            
            # 정보 없음 여부 확인
            has_info = "정보 없음" not in section_content.lower() and "확인할 수 없음" not in section_content.lower()
            
            # 섹션별 청크 생성
            section_type = section_title.lower().replace(" ", "_")
            
            # 문서, 메타데이터, ID 추가
            documents.append(f"{section_title}\n{section_content}")
            metadatas.append({
                "agent_type": "market_agent", 
                "company_name": company_name,
                "content_type": "section",
                "section_title": section_title,
                "section_type": section_type,
                "section_index": str(i),
                "has_info": str(has_info)
            })
            ids.append(f"market_analysis_{company_name}_{section_type}")
        
        # ChromaDB에 저장
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[시장 분석 에이전트] {company_name} 기업 시장 분석 결과 {len(documents)}개 청크로 저장 완료")
    except Exception as e:
        print(f"시장 분석 결과 저장 중 오류 발생: {e}")


# 테스트 코드
if __name__ == "__main__":
    # 테스트할 회사 선택
    test_company = "레브잇(올웨이즈)"
    
    # 시장 분석 실행
    result = analyze_market_position(test_company)
    
    # 결과 출력
    print("\n==== 시장 분석 결과 ====\n")
    print(result["market_analysis"])
    
    # 파일로 저장 (선택사항)
    with open(f"./market_analysis_output/{test_company}_market_analysis.txt", "w", encoding="utf-8") as f:
        f.write(result["market_analysis"])
    print(f"\n결과가 {test_company}_market_analysis.txt 파일에 저장되었습니다.")
