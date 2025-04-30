"""
PlaywrightAPI 기반 크롤러 - 셀레니움 대체 방식
"""

import asyncio
import pandas as pd
from datetime import datetime
import sys
import os

# 필요한 라이브러리가 설치되어 있는지 확인
try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright가 설치되어 있지 않습니다. 설치 중...")
    os.system("pip install playwright")
    os.system("playwright install")
    from playwright.async_api import async_playwright

class PlaywrightCrawler:
    def __init__(self, output_path="startup_list.csv"):
        """
        크롤러 초기화
        
        Args:
            output_path (str): 결과를 저장할 파일 경로
        """
        self.base_url = "https://www.innoforest.co.kr/dataroom"
        self.output_path = output_path
        self.companies = []
    
    def build_url(self, biz_array=21, page=1, invst_cd_array=32):
        """
        크롤링할 URL 생성
        
        Args:
            biz_array (int): 카테고리 ID (21: AI,딥테크,블록체인)
            page (int): 페이지 번호
            invst_cd_array (int): 투자단계 ID (32: series B)
            
        Returns:
            str: 크롤링할 URL
        """
        return f"{self.base_url}?bizArray={biz_array}&page={page}&invstCdArray={invst_cd_array}"
    
    async def crawl_page(self, page_obj, url):
        """
        특정 URL의 페이지 크롤링
        
        Args:
            page_obj: Playwright 페이지 객체
            url (str): 크롤링할 URL
            
        Returns:
            list: 기업 이름 리스트
        """
        try:
            print(f"페이지 요청 중: {url}")
            
            # 페이지 로드
            await page_obj.goto(url, wait_until="domcontentloaded")
            
            # 페이지가 완전히 로드될 때까지 대기
            await page_obj.wait_for_selector("table.css-8sdmb tbody tr", timeout=10000)
            
            # 기업 이름 추출
            companies = await page_obj.evaluate("""
                () => {
                    const companies = [];
                    const rows = document.querySelectorAll('table.css-8sdmb tbody tr');
                    
                    for (const row of rows) {
                        const nameElement = row.querySelector('td.cs-td-0 h4.corp p');
                        if (nameElement) {
                            companies.push(nameElement.textContent.trim());
                        }
                    }
                    
                    return companies;
                }
            """)
            
            # 결과 출력
            for company in companies:
                print(f"기업 추출: {company}")
                
            return companies
            
        except Exception as e:
            print(f"페이지 크롤링 중 오류: {e}")
            return []
    
    async def crawl_multiple_pages(self, pages=[1]):
        """
        여러 페이지 크롤링
        
        Args:
            pages (list): 크롤링할 페이지 번호 리스트
            
        Returns:
            bool: 성공 여부
        """
        async with async_playwright() as p:
            try:
                # 브라우저 실행
                print("브라우저 시작 중...")
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
                )
                
                page = await context.new_page()
                
                # 각 페이지 크롤링
                for page_num in pages:
                    url = self.build_url(page=page_num)
                    page_companies = await self.crawl_page(page, url)
                    
                    if page_companies:
                        self.companies.extend(page_companies)
                    
                    # 페이지 간 대기
                    await asyncio.sleep(2)
                
                # 브라우저 종료
                await browser.close()
                
                if not self.companies:
                    print("어떤 기업도 찾지 못했습니다.")
                    return False
                
                # 결과 저장
                self.save_results()
                return True
                
            except Exception as e:
                print(f"크롤링 중 오류: {e}")
                return False
    
    def save_results(self):
        """크롤링 결과를 CSV 파일로 저장"""
        try:
            df = pd.DataFrame({"기업명": self.companies})
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            print(f"결과가 {self.output_path}에 저장되었습니다.")
            return True
        except Exception as e:
            print(f"결과 저장 중 오류: {e}")
            return False
    
    def get_results(self):
        """크롤링 결과 반환"""
        return self.companies

def print_banner():
    """프로그램 배너 출력"""
    banner = """
    ========================================================
     혁신의 숲 스타트업 리스트 크롤러 (Playwright 버전)
    --------------------------------------------------------
     - 카테고리: AI,딥테크,블록체인
     - 최종투자단계: series B
     - 정렬: 기업 조회수 (최근 7일)
    ========================================================
    """
    print(banner)

async def main_async():
    """비동기 메인 함수"""
    # 현재 시간을 포함한 출력 파일명 생성
    output_path = f"startup_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"결과 저장 경로: {output_path}")
    
    # 크롤러 생성 및 실행
    crawler = PlaywrightCrawler(output_path=output_path)
    success = await crawler.crawl_multiple_pages(pages=[1])
    
    if success:
        results = crawler.get_results()
        print("\n크롤링된 기업 목록:")
        for i, company in enumerate(results, 1):
            print(f"{i}. {company}")
        print(f"\n총 {len(results)}개 기업이 크롤링되었습니다.")
        print(f"결과가 '{output_path}'에 저장되었습니다.")
        return 0
    else:
        print("크롤링에 실패했습니다.")
        return 1

def main():
    """메인 함수"""
    print_banner()
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
