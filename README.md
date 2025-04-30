# AI Startup Investment Evaluation Agent
본 프로젝트는 AI 활용 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview

- **Objective**: AI 스타트업의 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석
- **Method**: AI Agent + Agentic RAG (LangGraph 기반 워크플로우)
- **Target**: 시리즈 B 스타트업 (시장 검증이 완료되어 확장 단계에 있는 기업)

## Features

- 스타트업 데이터 자동 수집 및 분석 (혁신의숲, 뉴스 기사 등)
- PDF 자료 기반 정보 추출 및 OCR 처리
- 다중 에이전트 기반 종합 분석 (기술력, 재무, 성장성 등)
- 투자 기준별 판단 분류 및 근거 제시
- 투자 보고서 자동 생성 (투자추천/투자보류)
- 진행 상황 시각화 및 로깅

## Tech Stack 

| Category   | Details                      |
|------------|------------------------------|
| Framework  | LangGraph, LangChain, Python |
| LLM        | GPT-4o-mini via OpenAI API   |
| VectorDB   | ChromaDB                     |
|임베딩      | HuggingFace (ko-sroberta-multitask)|
| Data       | Playwright, PDF OCR(PyMuPDFLoader), Web Search(Tavily) |

## Agents

- **스타트업_탐색**: 시리즈 B 스타트업 10개 탐색 및 투자 후보 선정
- **스타트업_정보_수집**: 선정된 스타트업 정보 수집 및 PDF OCR 처리
- **기술_탐색_요약**: 스타트업 핵심 기술 탐색 및 요약 분석
- **시장_분석**: 스타트업의 시장 포지셔닝, 경쟁사 분석, 경쟁 우위, 성장성 및 위협 요소 분석
- **투자_판단**: 수집된 정보 기반 투자 추천/보류 결정 및 근거 제시
- **보고서_생성**: 투자 보고서 또는 투자보류 종합 보고서 생성

## Architecture

```
                      ┌───────────────┐
                      │  스타트업_선택   │◄─────────────┐
                      └───────┬───────┘              │
                              │                      │
                              ▼                      │
                 ┌────────────────────────┐          │
                 │   스타트업_정보_수집     │          │
                 └────────────┬───────────┘          │
                              │                      │
                              ▼                      │
                 ┌────────────────────────┐          │
                 │    기술_탐색_요약       │          │
                 └────────────┬───────────┘          │
                              │                      │
                              ▼                      │
                 ┌────────────────────────┐          │
                 │      시장_분석         │          │
                 └────────────┬───────────┘          │
                              │                      │
                              ▼                      │
                 ┌────────────────────────┐          │
                 │      투자_판단         │          │
                 └────────────┬───────────┘          │
                              │                      │
             ┌───────────────┴────────────────┐      │
             │                                │      │
             ▼                                ▼      │
┌────────────────────────┐       ┌─────────────────────────┐
│      투자추천          │       │     투자보류           │
└──────────┬─────────────┘       └─────────────────────────┘
           │                                   
           ▼                                   
┌────────────────────────┐                     
│      보고서_생성        │                    
└──────────┬─────────────┘                  
           │                                   
           ▼                                   
┌────────────────────────┐                     
│        완료            │                     
└────────────────────────┘                        
```

## Data Flow
1. 스타트업 크롤링 → 데이터 수집 → 벡터 저장소
2. 정보 추출 → 구조화 → 벡터 저장소
3. 기술 분석 → 웹 검색 → 요약 → 벡터 저장소
4. 시장 분석 → 경쟁사 비교 → 벡터 저장소
5. 투자 판단 → 종합 분석 → 의사결정
6. 보고서 생성 → 마크다운 문서

## Vector Database Schema

ChromaDB를 활용한 메타데이터 구조:

- 공통 필드:
    - company_name: 스타트업 이름 (모든 문서 공통)
    - agent_type: 에이전트 유형 (startup_agent, tech_agent, market_agent, invest_agent)

에이전트별 메타데이터:
1. startup_agent:
    - status: PDF 존재 여부 (0/1)
    - text_path: 추출된 텍스트 파일 경로
    - reason: PDF 없는 경우 이유 기록
2. tech_agent:
    - summary: 기술 요약 정보
    - confidence: 정보 신뢰도 점수 (0.0-1.0)
    - urls: 참조된 정보 소스 URL들

3. market_agent:
    - content_type: 분석 유형 (full_analysis, summary 등)
    - has_sufficient_data: 충분한 시장 데이터 유무
    - competitors: 주요 경쟁사 목록

4. invest_agent:
    - decision: 투자 결정 (투자추천/투자보류)
    - summary: 결정 요약 근거

## Directory Structure

```
.
├── agents/
│   ├── company_info_agent.py  # PDF 정보 추출 및 웹 검색
│   ├── tech_agent.py         # 기술 탐색 및 요약
│   ├── market_agent.py       # 시장 분석 및 경쟁사 정보 수집
│   ├── invest_agent.py       # 투자 결정 판단
│   ├── report_agent.py       # 최종 보고서 생성
│   └── startup_agent.py      # 스타트업 크롤링
├── app.py                    # 메인 워크플로우 애플리케이션
├── chroma_store/             # 벡터 DB 저장소
├── data/                     # 스타트업 PDF 문서 저장
│   └── outputs/              # 정보 추출 결과
├── market_analysis_output/   # 시장 분석 결과 저장
├── progress/                 # 워크플로우 진행 상태 저장
├── prompts/                  # 프롬프트 템플릿
├── reports/                  # 생성된 보고서 저장
├── startup_list_*.csv        # 크롤링된 스타트업 목록
├── visualizations/           # 워크플로우 시각화 이미지
├── workflow.log              # 로그 파일
├── .env                      # 환경 변수(API 키 등)
└── README.md                 # 프로젝트 설명서
```

## Contributors 
- **구동빈**: 스타트업 정보 수집 에이전트
- **김경아**: 스타트업 탐색 에이전트
- **박정의**: 기술 탐색 및 핵심 기술 요약 에이전트
- **배진환**: 투자 판단 에이전트
- **이원행**: 시장 분석 및 보고서 생성 에이전트, LangGraph
- **정현섭**: 프로젝트 코디네이션 및 구조 설계
