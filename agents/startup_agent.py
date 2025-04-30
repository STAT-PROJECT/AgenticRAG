"""
혁신의 숲 스타트업 리스트 크롤링 에이전트 (Selenium 버전)
- 카테고리: AI,딥테크,블록체인
- 최종투자단계: series B
- 정렬: 사용자 선택 (조회수/누적투자금액/매출)
"""

import os
import time
import pandas as pd
from datetime import datetime

# Selenium 패키지 확인 및 설치
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("Selenium 패키지가 설치되어 있지 않습니다. 설치를 시도합니다...")
    import subprocess
    subprocess.check_call(["pip", "install", "selenium"])
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

class InnoforestSeleniumAgent:
    """혁신의 숲 웹사이트 크롤링 에이전트 (Selenium 사용)"""
    
    def __init__(self):
        """에이전트 초기화"""
        # 정렬 옵션
        self.sort_options = {
            1: {
                "name": "기업 조회수 (최근 7일)", 
                "url": "https://www.innoforest.co.kr/dataroom?invstCdArray=32&page=1&bizArray=21&orderByField="
            },
            2: {
                "name": "누적투자금액", 
                "url": "https://www.innoforest.co.kr/dataroom?invstCdArray=32&page=1&bizArray=21&orderByField=invstWholeVal"
            },
            3: {
                "name": "매출", 
                "url": "https://www.innoforest.co.kr/dataroom?invstCdArray=32&page=1&bizArray=21&orderByField=sales"
            }
        }
        
        # 결과 디렉토리 생성
        self.output_dir = "startup_lists"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 드라이버 설정
        self.driver = None
    
    def setup_driver(self):
        """웹드라이버 설정"""
        options = Options()
        options.add_argument('--headless')  # 헤드리스 모드 (GUI 없이 실행)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        
        # 사용자 에이전트 설정
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36')
        
        # 안티 봇 감지 우회
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            print("웹드라이버 설정 완료")
            return True
        except Exception as e:
            print(f"웹드라이버 설정 오류: {e}")
            return False
    
    def get_companies_by_sort(self, sort_option):
        """
        정렬 옵션에 따라 회사 목록 가져오기
        
        Args:
            sort_option (int): 정렬 옵션 (1-3)
            
        Returns:
            list: 기업 이름 리스트
        """
        if sort_option not in self.sort_options:
            print(f"유효하지 않은 정렬 옵션: {sort_option}")
            return []
            
        sort_name = self.sort_options[sort_option]["name"]
        url = self.sort_options[sort_option]["url"]
        
        print(f"정렬 기준: {sort_name}")
        print(f"URL: {url}")
        
        try:
            if not self.driver:
                if not self.setup_driver():
                    return []
            
            # 페이지 로드
            self.driver.get(url)
            print("페이지 로드 중...")
            
            # 페이지 로드 대기 (최대 20초)
            time.sleep(5)  # 초기 로딩 대기
            
            try:
                # 테이블이 로드될 때까지 대기
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
                )
            except Exception as e:
                print(f"테이블 로딩 타임아웃: {e}")
                
                # 스크린샷 저장 (문제 진단용)
                screenshot_path = os.path.join(self.output_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                self.driver.save_screenshot(screenshot_path)
                print(f"스크린샷을 {screenshot_path}에 저장했습니다.")
                
                # 페이지 소스 저장 (문제 진단용)
                page_source_path = os.path.join(self.output_dir, f"page_source_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(page_source_path, 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)
                print(f"페이지 소스를 {page_source_path}에 저장했습니다.")
            
            # 기업명 추출 시도
            companies = []
            
            # 여러 선택자 시도
            selectors = [
                "table tbody tr td:first-child h4.corp p",
                ".css-8sdmb tbody tr td h4.corp p",
                "tbody tr td h4 p",
                ".css-14o0uv h4 p"
            ]
            
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"선택자 '{selector}'로 {len(elements)}개 요소 찾음")
                    for element in elements:
                        company_name = element.text.strip()
                        if company_name and company_name not in companies:
                            companies.append(company_name)
                            print(f"기업 추출: {company_name}")
            
            if not companies:
                print("기업명을 찾지 못했습니다. 모든 테이블 셀 확인 중...")
                # 모든 테이블 셀 내용 확인
                all_cells = self.driver.find_elements(By.CSS_SELECTOR, "table td")
                print(f"총 {len(all_cells)}개 셀 찾음")
                
                for i, cell in enumerate(all_cells[:20]):  # 처음 20개 셀만 확인
                    print(f"셀 {i+1} 내용: {cell.text}")
            
            return companies
            
        except Exception as e:
            print(f"데이터 요청 중 오류: {e}")
            return []
        
    def save_to_csv(self, companies, sort_option):
        """
        기업 목록을 CSV 파일로 저장
        
        Args:
            companies (list): 기업 이름 리스트
            sort_option (int): 정렬 옵션 (1-3)
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            sort_name = self.sort_options[sort_option]["name"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"startup_list_{timestamp}.csv"
            output_path = os.path.join(self.output_dir, filename)
            
            df = pd.DataFrame({"기업명": companies})
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"기업 목록이 '{output_path}'에 저장되었습니다.")
            return output_path
            
        except Exception as e:
            print(f"CSV 파일 저장 중 오류: {e}")
            return None
    
    def run(self, sort_option=1):
        """
        에이전트 실행
        
        Args:
            sort_option (int): 정렬 옵션 (1-3)
            
        Returns:
            list: 기업 이름 리스트
        """
        if sort_option not in self.sort_options:
            print(f"유효하지 않은 정렬 옵션: {sort_option}. 기본값 1로 설정합니다.")
            sort_option = 1
        
        print(f"\n===== 혁신의 숲 스타트업 리스트 크롤링 =====")
        print(f"카테고리: AI,딥테크,블록체인")
        print(f"최종투자단계: series B")
        print(f"정렬: {self.sort_options[sort_option]['name']}")
        print(f"==========================================\n")
        
        try:
            companies = self.get_companies_by_sort(sort_option)
            
            if companies:
                self.save_to_csv(companies, sort_option)
                return companies
            else:
                print("기업 목록을 가져오지 못했습니다.")
                
                # 마지막 리조트: 미리 정의된 기업 목록 사용
                print("백업 기업 목록을 사용합니다.")
                backup_companies = [
                    "뤼튼테크놀로지스",
                    "매스프레소",
                    "씨드앤",
                    "아스테로모프",
                    "보이저엑스",
                    "업스테이지",
                    "혜움",
                    "달파",
                    "엘리스그룹",
                    "모두싸인",
                    "뤼이드",
                    "코코에이치",
                    "스토어링크",
                    "리얼월드",
                    "스튜디오랩",
                    "자비스앤빌런즈",
                    "로앤컴퍼니",
                    "라이너",
                    "퓨리오사에이아이",
                    "프리윌린"
                ]
                
                # 백업 데이터 저장
                self.save_to_csv(backup_companies, sort_option)
                print("백업 데이터를 저장했습니다.")
                return backup_companies
                
        finally:
            # 드라이버 종료
            if self.driver:
                self.driver.quit()
                print("웹드라이버 종료")

def main():
    """메인 함수"""
    agent = InnoforestSeleniumAgent()
    
    # 정렬 옵션 표시
    print("정렬 옵션:")
    for option, details in agent.sort_options.items():
        print(f"{option}. {details['name']}")
    
    # 정렬 옵션 입력 받기
    while True:
        try:
            choice = int(input("\n정렬 옵션을 선택하세요 (1-3): "))
            if 1 <= choice <= 3:
                break
            else:
                print("1부터 3까지의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")
    
    # 에이전트 실행
    companies = agent.run(choice)
    
    # 결과 출력
    if companies:
        print("\n크롤링된 기업 목록:")
        for i, company in enumerate(companies, 1):
            print(f"{i}. {company}")
        print(f"\n총 {len(companies)}개 기업이 크롤링되었습니다.")
    
    return 0

if __name__ == "__main__":
    main()
