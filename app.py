import os
from typing import TypedDict, List, Dict, Optional, Literal, Any
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import chromadb

# 각 에이전트 모듈 임포트
from agents.company_info_agent import collect_startup_info
from agents.tech_agent import tech_exploration
from agents.market_agent import analyze_market_position
from agents.invest_agent import investment_decision_agent
from agents.report_agent import report_generation_agent

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    filename="workflow.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 상태 타입 정의 (TypedDict)
class AgentState(TypedDict):
    current_step: str
    startup_list: List[Dict]
    selected_startup: Optional[Dict]
    startup_info: Optional[Dict]
    tech_info: Optional[Dict]
    market_info: Optional[Dict]
    investment_decision: Optional[Literal["투자추천", "투자보류"]]
    report: Optional[str]
    all_investment_decisions: Dict[str, Literal["투자추천", "투자보류"]]
    processed_startups_count: int
    total_startups_count: int
    enable_parallel: bool  # 병렬 처리 활성화 옵션
    messages: List[Dict]

# 진행 상황 저장 함수
def save_progress(state: AgentState, step_name: str) -> AgentState:
    """진행 상황 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"progress/{step_name}_{timestamp}.json"
    
    # 디렉토리 생성
    os.makedirs("progress", exist_ok=True)
    
    # 상태를 JSON으로 변환 (복잡한 객체 제외)
    serializable_state = {k: v for k, v in state.items() 
                         if isinstance(v, (dict, list, str, int, float, bool)) or v is None}
    
    # 파일에 저장
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, ensure_ascii=False, indent=2)
    
    logging.info(f"진행 상황 저장됨: {filename}")
    return state

# 워크플로우 노드 구현
def startup_selection(state: AgentState) -> AgentState:
    """스타트업 선택 노드"""
    # 테스트를 위한 더미 데이터
    startup_list = [
        {"name": "씨드앤", "sector": "AI"},
        {"name": "레브잇", "sector": "이커머스"},
        {"name": "화이트큐브", "sector": "핀테크"}
    ]
    
    if not state["selected_startup"] and state["startup_list"]:
        # 리스트의 첫번째 스타트업 선택
        selected_startup = state["startup_list"][0]
        updated_state = {
            **state,
            "current_step": "스타트업_선택",
            "selected_startup": selected_startup,
            "messages": state["messages"] + [{"role": "system", "content": f"{selected_startup['name']} 스타트업이 선택되었습니다."}]
        }
    elif not state["selected_startup"]:
        # 리스트에 스타트업이 없으면 기본 리스트 사용
        updated_state = {
            **state,
            "current_step": "스타트업_선택",
            "startup_list": startup_list,
            "selected_startup": startup_list[0],
            "messages": state["messages"] + [{"role": "system", "content": f"{startup_list[0]['name']} 스타트업이 선택되었습니다."}]
        }
    else:
        updated_state = state
        
    return save_progress(updated_state, "startup_selection")

def router(state: AgentState) -> str:
    """다음 단계 라우터"""
    current_step = state["current_step"]
    logging.info(f"라우터: 현재 단계 '{current_step}'에서 다음 단계 결정")
    
    # 투자 결정에 따른 라우팅
    if current_step == "투자_판단":
        decision = state.get("investment_decision")
        if decision == "투자추천":
            return "보고서_생성"
        else:
            # 전체 프로세스 완료 여부 확인
            processed = state.get("processed_startups_count", 0)
            total = state.get("total_startups_count", 0)
            
            if processed >= total:
                return "완료"
            else:
                return "스타트업_선택"
    
    # 병렬 처리 분기
    if current_step == "스타트업_정보_수집":
        if state.get("enable_parallel", False):
            logging.info("병렬 처리 활성화: 기술+시장 병렬 분석 경로 선택")
            return "기술_및_시장_분석"
        else:
            return "기술_탐색_요약"
    
    # 일반 워크플로우 진행
    step_flow = {
        "스타트업_선택": "스타트업_정보_수집",
        "기술_탐색_요약": "시장_분석",
        "시장_분석": "투자_판단",
        "기술_및_시장_분석": "투자_판단",  # 병렬 처리 후 바로 투자 판단으로
        "보고서_생성": "완료"
    }
    
    return step_flow.get(current_step, "완료")

def process_startup_info(state: AgentState) -> AgentState:
    """스타트업 정보 수집 노드"""
    return collect_startup_info(state)  # 수정: state 전체를 전달

def process_tech_exploration(state: AgentState) -> AgentState:
    """기술 탐색 노드"""
    return tech_exploration(state)  # 수정: state 전체를 전달

def process_market_analysis(state: AgentState) -> AgentState:
    """시장 분석 노드"""
    return analyze_market_position(state)  # 수정: state 전체를 전달

def combine_tech_market_analysis(state: AgentState) -> AgentState:
    """기술 및 시장 분석을 병렬로 처리"""
    company = state["selected_startup"]["name"]
    
    try:
        # 병렬 실행을 위한 스레드 풀 사용
        with ThreadPoolExecutor(max_workers=2) as executor:
            # tech_exploration은 company_name만 받지 않고 state 전체를 받음
            tech_future = executor.submit(tech_exploration, state)
            
            # analyze_market_position도 state 전체를 받음
            market_state = {**state}
            market_future = executor.submit(analyze_market_position, market_state)
            
            # 결과는 각각 state 객체
            tech_result_state = tech_future.result()
            market_result_state = market_future.result()
            
            tech_info = tech_result_state.get("tech_info")
            market_info = market_result_state.get("market_info")
        
        # 결과를 상태에 병합
        updated_state = {
            **state,
            "current_step": "기술_및_시장_분석",
            "tech_info": tech_info,
            "market_info": market_info,
            "messages": state["messages"] + [{"role": "system", "content": f"{company} 기술 및 시장 분석 완료 (병렬 처리)"}]
        }
        
        return save_progress(updated_state, "parallel_analysis")
    except Exception as e:
        logging.error(f"병렬 처리 실패: {str(e)}")
        # 에러 발생시 기본 상태 반환
        return {
            **state,
            "messages": state["messages"] + [{"role": "system", "content": f"병렬 처리 중 오류: {str(e)}"}]
        }

def process_investment_decision(state: AgentState) -> AgentState:
    """투자 판단 노드"""
    return investment_decision_agent(state)  # 수정: state 전체를 전달

def generate_report(state: AgentState) -> AgentState:
    """보고서 생성 노드"""
    return report_generation_agent(state)  # 수정: state 전체를 전달

# 워크플로우 그래프 구성
def create_agent_workflow():
    # 초기 상태 정의
    initial_state = {
        "current_step": "스타트업_선택",
        "startup_list": [],
        "selected_startup": None,
        "startup_info": None, 
        "tech_info": None,
        "market_info": None,
        "investment_decision": None,
        "report": None,
        "all_investment_decisions": {},
        "processed_startups_count": 0,
        "total_startups_count": 3,
        "enable_parallel": True,  # 병렬 처리 기본 활성화
        "messages": []
    }
    
    # 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 에러 핸들러 정의
    def error_handler(state: AgentState, exception: Exception) -> AgentState:
        """에러 처리 로직"""
        error_msg = f"⚠️ 에러 발생: {str(exception)}"
        print(error_msg)
        logging.error(error_msg)
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "system", "content": error_msg}
            ]
        }
    
    # 정보 완전성 확인 함수 (함수를 create_agent_workflow 내부로 이동)
    def check_info_completeness(state: AgentState) -> str:
        """정보 완전성 확인 후 분기 또는 회귀 결정"""
        logging.info("정보 완전성 검증 중...")
        
        if not state.get("tech_info") or state["tech_info"].get("confidence", 0) < 0.7:
            logging.info("기술 정보 부족: 기술 탐색 단계로 회귀")
            return "기술_탐색_요약"
        
        market_info = state.get("market_info", {})
        if market_info.get("has_sufficient_data") is False:
            logging.info("시장 정보 부족: 시장 분석 단계로 회귀")
            return "시장_분석"
        
        decision = state.get("investment_decision")
        if decision == "투자추천":
            logging.info("투자 추천: 보고서 생성 단계로 진행")
            return "보고서_생성"
        else:
            logging.info("투자 보류: 다음 스타트업 선택 또는 종료")
            return "투자보류"
    
    # 노드 추가
    workflow.add_node("스타트업_선택", startup_selection)
    workflow.add_node("스타트업_정보_수집", process_startup_info)
    workflow.add_node("기술_탐색_요약", process_tech_exploration)
    workflow.add_node("기술_및_시장_분석", combine_tech_market_analysis)
    workflow.add_node("시장_분석", process_market_analysis)
    workflow.add_node("투자_판단", process_investment_decision)
    workflow.add_node("보고서_생성", generate_report)
    
    # 엣지 설정
    workflow.add_edge("스타트업_선택", "스타트업_정보_수집")

    # 병렬 처리 분기
    workflow.add_conditional_edges(
        "스타트업_정보_수집",
        lambda s: "기술_및_시장_분석" if s.get("enable_parallel", False) else "기술_탐색_요약",
        {
            "기술_및_시장_분석": "기술_및_시장_분석",
            "기술_탐색_요약": "기술_탐색_요약"
        }
    )

    workflow.add_edge("기술_탐색_요약", "시장_분석")
    workflow.add_edge("시장_분석", "투자_판단")
    workflow.add_edge("기술_및_시장_분석", "투자_판단")

    # 투자 판단 후 정보 완전성 검증 및 분기
    workflow.add_conditional_edges(
        "투자_판단",
        check_info_completeness,
        {
            "기술_탐색_요약": "기술_탐색_요약",
            "시장_분석": "시장_분석",
            "보고서_생성": "보고서_생성",
            "투자보류": "스타트업_선택"  # 경로 목적지 설정
        }
    )

    workflow.add_edge("보고서_생성", END)
    
    # 시작 노드 설정
    workflow.set_entry_point("스타트업_선택")
    
    # 컴파일된 그래프 반환
    return workflow.compile()

def visualize_workflow(workflow, state):
    """워크플로우 그래프 시각화"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # 그래프 추출 (LangGraph 내부 그래프 구조에 접근)
        G = nx.DiGraph()
        for node in workflow.nodes:
            G.add_node(node)
        
        for edge in workflow.edges:
            G.add_edge(edge[0], edge[1])
        
        # 레이아웃 및 그리기
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # 일관된 시각화를 위해 시드 고정
        
        # 현재 노드 강조
        current = state["current_step"]
        node_colors = ['red' if n == current else 'lightblue' for n in G.nodes()]
        
        # 그래프 그리기
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=2000, font_size=10, font_weight='bold')
        
        # 진행 경로 강조 (선택사항)
        if "messages" in state:
            steps = [msg["content"].split()[0] for msg in state["messages"] 
                     if msg["role"] == "system" and "완료" in msg["content"]]
            if len(steps) > 1:
                path_edges = [(steps[i], steps[i+1]) for i in range(len(steps)-1) 
                              if (steps[i], steps[i+1]) in G.edges]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                       width=2.0, edge_color='red')
        
        # 저장 및 표시
        os.makedirs("visualizations", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualizations/workflow_{current}_{timestamp}.png"
        plt.savefig(filename)
        print(f"📊 워크플로우 시각화 이미지 저장됨: {filename}")
        
    except Exception as e:
        print(f"시각화 중 오류 발생: {e}")
        logging.error(f"시각화 실패: {str(e)}")

# 메인 실행 함수
def main():
    print("🚀 스타트업 투자 평가 에이전트 시스템 시작")
    logging.info("애플리케이션 시작")
    
    # 필요한 디렉토리 생성
    os.makedirs("progress", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # 워크플로우 생성
    workflow = create_agent_workflow()
    
    # 초기 상태 설정
    config = {
        "current_step": "스타트업_선택",
        "startup_list": [
            {"name": "씨드앤", "sector": "AI"},
            {"name": "레브잇", "sector": "이커머스"},
            {"name": "화이트큐브", "sector": "핀테크"}
        ],
        "selected_startup": None,
        "startup_info": None,
        "tech_info": None,
        "market_info": None,
        "investment_decision": None,
        "report": None,
        "all_investment_decisions": {},
        "processed_startups_count": 0,
        "total_startups_count": 3,
        "enable_parallel": True,  # 병렬 처리 활성화
        "messages": []
    }
    
    # 에이전트 실행
    for step in workflow.stream(config):
        current_node = step.get("current_node")
        state = step.get("state", {})
        
        if current_node:
            current_step = state.get("current_step", "Unknown")
            print(f"\n📍 현재 단계: {current_step} (노드: {current_node})")
            logging.info(f"실행 중: {current_node} (단계: {current_step})")
            
            # 주요 단계에서 시각화 실행
            if current_node in ["스타트업_선택", "투자_판단", "보고서_생성", "기술_및_시장_분석"]:
                visualize_workflow(workflow, state)
            
            # 메시지 로그 출력
            messages = state.get("messages", [])
            if messages and len(messages) > 0:
                latest_message = messages[-1]
                print(f"🔔 {latest_message['role']}: {latest_message['content']}")
            
            # 투자 결정 출력
            if current_node == "투자_판단":
                decision = state.get("investment_decision")
                if decision:
                    print(f"💰 투자 결정: {decision}")
                    logging.info(f"투자 결정: {state['selected_startup']['name']} -> {decision}")
            
            # 보고서 완료 시 내용 일부 출력
            if current_node == "보고서_생성" and state.get("report"):
                report = state.get("report")
                print(f"📄 보고서 생성 완료 (일부 내용):\n{report[:200]}...\n")
                
                # 보고서 저장
                company = state["selected_startup"]["name"]
                os.makedirs("reports", exist_ok=True)
                with open(f"reports/{company}_투자보고서.md", "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"💾 보고서 저장 완료: reports/{company}_투자보고서.md")
    
    # 최종 결과 출력
    all_decisions = workflow.get_state().get("all_investment_decisions", {})
    print("\n✅ 투자 평가 완료")
    print("📊 투자 결정 요약:")
    for company, decision in all_decisions.items():
        print(f" - {company}: {decision}")
    
    # 종합 결과 저장
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decisions": all_decisions,
        "total_processed": len(all_decisions)
    }
    
    with open("investment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("📝 투자 결정 요약 저장 완료: investment_summary.json")
    logging.info("애플리케이션 종료")

if __name__ == "__main__":
    main()