#!/usr/bin/env python3
"""
검색 메타데이터 생성 시스템
- LLM을 사용한 키워드 추출
- ScienceON API를 통한 문서 검색
- 구조화된 JSON 파일 생성
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import google.generativeai as genai

# 기존 ScienceON API 클라이언트 import
from scienceon_api_example import ScienceONAPIClient

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KeywordExtractor:
    """LLM을 사용한 키워드 추출기"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        키워드 추출기 초기화
        
        Args:
            api_key: Google API 키
            model_name: 사용할 Gemini 모델명
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = self._init_gemini()
        
    def _init_gemini(self) -> genai.GenerativeModel:
        """Gemini 모델 초기화"""
        genai.configure(api_key=self.api_key)
        generation_config = genai.GenerationConfig(
            temperature=0.2,
            candidate_count=1,
        )
        
        return genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )
    
    def _create_keyword_prompt(self, query: str) -> str:
        """키워드 추출을 위한 프롬프트 생성"""
        prompt = f"""당신은 논문 검색을 위한 키워드 추출 전문가입니다. 주어진 질문에서 ScienceON API 검색에 최적화된 핵심 키워드들을 한국어와 영어로 각각 추출해주세요.

질문: "{query}"

다음 형식으로 키워드를 추출하세요:

1. 한국어 키워드: 3-5개의 핵심 키워드를 쉼표로 구분 (전자교과서)
2. 영어 키워드: 위 한국어 키워드들의 영어 번역을 쉼표로 구분

규칙:
- 전문용어와 기술용어를 우선적으로 선택
- 축약어가 있으면 축약어 사용 (예: AI, ML, NLP)
- 각 키워드는 1-20자 이내로 간결하게


출력 형식:
한국어: 키워드1, 키워드2, 키워드3, 키워드4
영어: keyword1, keyword2, keyword3, keyword4



키워드:"""
        return prompt
    
    def extract_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        질문에서 키워드 추출 (한국어, 영어 각각)
        
        Args:
            query: 추출할 질문
            
        Returns:
            {'korean': [키워드들], 'english': [키워드들]} 형태의 딕셔너리
        """
        try:
            prompt = self._create_keyword_prompt(query)
            response = self.model.generate_content(prompt)
            
            # 응답에서 키워드 추출
            response_text = response.text.strip()
            
            # "키워드:" 텍스트 제거
            if response_text.startswith("키워드:"):
                response_text = response_text[4:].strip()
            
            # 한국어와 영어 키워드 분리
            korean_keywords = []
            english_keywords = []
            
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('한국어:'):
                    korean_text = line.replace('한국어:', '').strip()
                    korean_keywords = [kw.strip() for kw in korean_text.split(',') if kw.strip()]
                elif line.startswith('영어:'):
                    english_text = line.replace('영어:', '').strip()
                    english_keywords = [kw.strip() for kw in english_text.split(',') if kw.strip()]
            
            # 빈 키워드 제거 및 길이 제한
            korean_keywords = [kw for kw in korean_keywords if kw and 1 < len(kw) <= 30]
            english_keywords = [kw for kw in english_keywords if kw and 1 < len(kw) <= 30]
            
            result = {
                'korean': korean_keywords,
                'english': english_keywords
            }
            
            logging.info(f"질문: {query}")
            logging.info(f"한국어 키워드: {korean_keywords}")
            logging.info(f"영어 키워드: {english_keywords}")
            
            return result
            
        except Exception as e:
            logging.error(f"키워드 추출 실패: {e}")
            return {'korean': [], 'english': []}


class SearchQueryGenerator:
    """Gemini를 활용한 지능적 검색어 생성기"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        검색어 생성기 초기화
        
        Args:
            api_key: Google Gemini API 키
            model_name: 사용할 Gemini 모델명
        """
        self.model = self._init_gemini(api_key, model_name)
    
    def _init_gemini(self, api_key: str, model_name: str):
        """Gemini 모델 초기화"""
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    
    def _create_search_query_prompt(self, query: str, keywords_dict: Dict[str, List[str]]) -> str:
        """검색어 생성을 위한 프롬프트 생성"""
        korean_keywords = keywords_dict.get('korean', [])
        english_keywords = keywords_dict.get('english', [])
        
        prompt = f"""당신은 ScienceON API 검색을 위한 전문가입니다. 주어진 질문과 키워드들을 바탕으로 효과적인 검색어들을 생성해주세요.

질문: "{query}"

한국어 키워드: {', '.join(korean_keywords)}
영어 키워드: {', '.join(english_keywords)}

다음 검색 연산자들을 활용하여 검색어를 생성하세요:

1. 공백 연산자: 두 개 이상의 검색어를 모두 포함하는 문서를 검색합니다.
   예: "나노 기계"

2. | 연산자: 두 개 이상의 검색어 중 1개 이상을 포함하는 문서를 검색합니다.
   예: "나노|기계"

3. * 연산자: 검색어 뒤 0개 이상의 임의의 문자가 포함된 문서를 검색합니다.
   예: "나노*"

4. ( ) 연산자: 괄호 안의 검색어가 우선순위로 지정됩니다.
   예: "나노 (기계 | machine)"

중요한 규칙:
- 질문의 핵심 의도와 가장 관련성 높은 검색어를 먼저 생성하세요
- 자연스러운 구문과 문맥을 고려한 검색어를 우선하세요
- 예: "AI mathematics machine learning textbook" 같은 자연스러운 구문
- 단순하고 실용적인 검색어를 우선하세요
- 복잡한 따옴표나 정확한 구문 검색은 피하세요
- 각 검색어는 한 줄에 하나씩 작성하세요
- 최대 12개의 검색어를 생성하세요
- 설명이나 번호 없이 검색어만 나열하세요

검색어 목록:"""
        return prompt
    
    def generate_search_queries(self, query: str, keywords_dict: Dict[str, List[str]]) -> List[str]:
        """
        Gemini를 활용하여 지능적인 검색어 생성
        
        Args:
            query: 원본 질문
            keywords_dict: {'korean': [...], 'english': [...]} 형태의 키워드 딕셔너리
            
        Returns:
            검색어 리스트
        """
        try:
            prompt = self._create_search_query_prompt(query, keywords_dict)
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # 검색어 파싱
            search_queries = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 설명적인 텍스트나 빈 줄 제외
                if (not line or 
                    line.startswith('검색어:') or 
                    line.startswith('예:') or
                    line.startswith('다음은') or
                    line.startswith('ScienceON') or
                    line.startswith('규칙:') or
                    line.startswith('질문:') or
                    line.startswith('한국어 키워드:') or
                    line.startswith('영어 키워드:') or
                    len(line) < 3):
                    continue
                
                # 번호나 기호 제거
                clean_line = line
                
                # 번호 패턴 제거 (1., 2., 10., 11. 등)
                import re
                clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                
                # 기호 제거
                if clean_line.startswith(('-', '•', '.', '`')):
                    clean_line = clean_line[1:].strip()
                
                # 백틱 제거
                clean_line = clean_line.replace('`', '').strip()
                
                # 따옴표 정리 - 원본 유지
                clean_line = clean_line.replace('"', '"').replace('"', '"')
                
                if clean_line and len(clean_line) <= 100 and not clean_line.startswith('다음은'):
                    search_queries.append(clean_line)
            
            # 중복 제거
            unique_queries = []
            seen = set()
            for query in search_queries:
                if query not in seen:
                    unique_queries.append(query)
                    seen.add(query)
            
            logging.info(f"Gemini가 생성한 검색어: {unique_queries}")
            return unique_queries[:15]  # 최대 15개로 제한
            
        except Exception as e:
            logging.error(f"검색어 생성 실패: {e}")
            # 실패 시 기본 키워드들 반환
            korean_keywords = keywords_dict.get('korean', [])
            english_keywords = keywords_dict.get('english', [])
            return korean_keywords + english_keywords


class SearchMetaGenerator:
    """검색 메타데이터 생성기 - 키워드 추출 + 문서 검색"""
    
    def __init__(self, gemini_api_key: str, scienceon_credentials_path: str = "./configs/scienceon_api_credentials.json"):
        """
        검색 메타데이터 생성기 초기화
        
        Args:
            gemini_api_key: Gemini API 키
            scienceon_credentials_path: ScienceON API 자격증명 파일 경로
        """
        self.keyword_extractor = KeywordExtractor(gemini_api_key)
        self.scienceon_client = ScienceONAPIClient(Path(scienceon_credentials_path))
        self.query_generator = SearchQueryGenerator(gemini_api_key)
    
    def _prepare_search_query_for_api(self, search_query: str) -> str:
        """
        검색어를 ScienceON API에 전달하기 위해 준비
        
        Args:
            search_query: 원본 검색어
            
        Returns:
            API용 검색어
        """
        # JSON에서 이스케이프된 따옴표를 원래 따옴표로 복원
        api_query = search_query.replace('\\"', '"')
        
        # 백슬래시가 포함된 경우 제거
        api_query = api_query.replace('\\', '')
        
        return api_query
    
    def process_query(self, query: str, min_documents: int = 50) -> Dict[str, Any]:
        """
        단일 쿼리 처리: 키워드 추출 + 문서 검색
        
        Args:
            query: 검색할 질문
            min_documents: 최소 보장 문서 수 (기본 50개)
            
        Returns:
            처리 결과 딕셔너리
        """
        logging.info(f"질문 처리 시작: {query[:50]}...")
        
        # 1. 키워드 추출
        keywords_dict = self.keyword_extractor.extract_keywords(query)
        korean_keywords = keywords_dict.get('korean', [])
        english_keywords = keywords_dict.get('english', [])
        
        # 2. 검색어 생성 (Gemini 활용)
        search_queries = self.query_generator.generate_search_queries(query, keywords_dict)
        logging.info(f"생성된 검색어 {len(search_queries)}개: {search_queries[:5]}...")  # 처음 5개만 로그
        
        # 3. 질문 언어 감지 및 우선 검색어 선택
        is_english_query = any(char.isascii() and char.isalpha() for char in query[:50])
        
        if is_english_query:
            logging.info(f"영어 질문 감지: 영어 우선 검색어 사용")
        else:
            logging.info(f"한국어 질문 감지: 한국어 우선 검색어 사용")
        
        # 4. 검색어로 문서 검색 (50개 이상 확보할 때까지 반복)
        all_documents = []
        page = 1
        
        while len(all_documents) < min_documents and page <= 2:
            logging.info(f"페이지 {page} 검색 중... (현재 {len(all_documents)}개 문서)")
            
            # 검색어로 검색 (페이지당 최대 10개 검색어)
            queries_per_page = search_queries[:10] if page == 1 else search_queries[10:20]
            
            for search_query in queries_per_page:
                # 검색어를 API에 전달할 때 따옴표 처리
                api_query = self._prepare_search_query_for_api(search_query)
                docs = self.scienceon_client.search_articles(api_query, cur_page=page, row_count=20)
                all_documents.extend(docs)
                
                if len(all_documents) >= min_documents:
                    break
            
            page += 1
        
        # 5. 중복 제거 (제목 기준)
        unique_docs = []
        seen_titles = set()
        for doc in all_documents:
            title = doc.get('title', '')
            if title and title not in seen_titles:
                # source 필드 제거 (ScienceON이 자명하므로)
                if 'source' in doc:
                    del doc['source']
                unique_docs.append(doc)
                seen_titles.add(title)
        
        # 6. 결과 정리
        result = {
            'question': query,
            'keywords': {
                'korean': korean_keywords,
                'english': english_keywords
            },
            'search_queries': search_queries,
            'documents': unique_docs,  # 모든 문서 포함
            'total_documents_found': len(unique_docs),
            'search_timestamp': datetime.now().isoformat()
        }
        
        total_keywords = len(korean_keywords) + len(english_keywords)
        logging.info(f"질문 처리 완료: {total_keywords}개 키워드 → {len(search_queries)}개 검색어 → {len(unique_docs)}개 문서")
        return result
    
    def process_queries(self, queries: List[str], min_documents_per_query: int = 50) -> List[Dict[str, Any]]:
        """
        여러 쿼리 일괄 처리
        
        Args:
            queries: 처리할 질문 리스트
            min_documents_per_query: 쿼리당 최소 보장 문서 수 (기본 50개)
            
        Returns:
            처리 결과 리스트
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logging.info(f"진행률: {i}/{len(queries)}")
            try:
                result = self.process_query(query, min_documents_per_query)
                results.append(result)
            except Exception as e:
                logging.error(f"쿼리 처리 실패: {query[:50]}... - {e}")
                # 실패한 쿼리도 결과에 포함
                results.append({
                    'question': query,
                    'keywords': [],
                    'documents': [],
                    'total_documents_found': 0,
                    'error': str(e),
                    'search_timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def save_results_to_json(self, results: List[Dict[str, Any]], output_path: str = None) -> str:
        """
        결과를 JSON 파일로 저장
        
        Args:
            results: 저장할 결과 리스트
            output_path: 출력 파일 경로 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"search_meta_results_{timestamp}.json"
        
        # 결과 요약 통계 추가
        summary = {
            'total_queries': len(results),
            'successful_queries': len([r for r in results if 'error' not in r]),
            'total_documents': sum(r.get('total_documents_found', 0) for r in results),
            'generation_timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logging.info(f"결과가 {output_path}에 저장되었습니다.")
        return output_path
    
    def close(self):
        """리소스 정리"""
        self.scienceon_client.close_session()

def test_keyword_extraction():
    """키워드 추출 테스트 함수"""
    
    # API 키 설정 (환경변수에서 가져오거나 직접 설정)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # configs 폴더에서 API 키 로드
        try:
            with open('./configs/gemini_api_credentials.json', 'r', encoding='utf-8') as f:
                credentials = json.load(f)
                api_key = credentials.get('api_key')
        except Exception as e:
            print(f"❌ API 키를 찾을 수 없습니다: {e}")
            return
    
    if not api_key:
        print("❌ GOOGLE_API_KEY 환경변수나 configs/gemini_api_credentials.json 파일에 API 키를 설정해주세요.")
        return
    
    # 키워드 추출기 초기화
    extractor = KeywordExtractor(api_key)
    
    # test.csv에서 질문 로드
    try:
        df = pd.read_csv('test.csv')
        print(f"📄 test.csv에서 {len(df)}개의 질문을 로드했습니다.")
        
        # 테스트할 질문 개수 설정 (처음 5개만 테스트)
        MAX_QUESTIONS = 5
        test_queries = df['Question'].head(MAX_QUESTIONS).tolist()
        
    except Exception as e:
        print(f"❌ test.csv 파일을 읽을 수 없습니다: {e}")
        print("기본 테스트 질문을 사용합니다.")
        test_queries = [
            "인공지능의 미래는 어떻게 될까요?",
            "머신러닝과 딥러닝의 차이점은 무엇인가요?"
        ]
    
    print("🔍 키워드 추출 테스트 시작")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 테스트 {i}: {query[:80]}...")
        
        keywords_dict = extractor.extract_keywords(query)
        korean_keywords = keywords_dict.get('korean', [])
        english_keywords = keywords_dict.get('english', [])
        
        # 검색어 생성 테스트
        query_generator = SearchQueryGenerator(api_key)
        search_queries = query_generator.generate_search_queries(query, keywords_dict)
        
        print(f"   🔑 한국어 키워드: {korean_keywords}")
        print(f"   🔑 영어 키워드: {english_keywords}")
        print(f"   🔍 생성된 검색어 ({len(search_queries)}개): {search_queries[:3]}...")
        print(f"   📊 총 키워드 개수: {len(korean_keywords) + len(english_keywords)}개")
    
    print("\n✅ 키워드 추출 테스트 완료!")

def test_full_search_pipeline():
    """전체 검색 파이프라인 테스트"""
    
    # API 키 설정
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        try:
            with open('./configs/gemini_api_credentials.json', 'r', encoding='utf-8') as f:
                credentials = json.load(f)
                gemini_api_key = credentials.get('api_key')
        except Exception as e:
            print(f"❌ Gemini API 키를 찾을 수 없습니다: {e}")
            return
    
    if not gemini_api_key:
        print("❌ GOOGLE_API_KEY 환경변수나 configs/gemini_api_credentials.json 파일에 API 키를 설정해주세요.")
        return
    
    # ScienceON API 자격증명 확인
    scienceon_credentials_path = './configs/scienceon_api_credentials.json'
    if not os.path.exists(scienceon_credentials_path):
        print(f"❌ ScienceON API 자격증명 파일을 찾을 수 없습니다: {scienceon_credentials_path}")
        return
    
    try:
        # 검색 메타데이터 생성기 초기화
        search_generator = SearchMetaGenerator(gemini_api_key, scienceon_credentials_path)
        
        # test.csv에서 질문 로드
        df = pd.read_csv('test.csv')
        print(f"📄 test.csv에서 {len(df)}개의 질문을 로드했습니다.")
        
        # 테스트할 질문 개수 설정 (처음 3개만 테스트)
        MAX_QUESTIONS = 3
        test_queries = df['Question'].head(MAX_QUESTIONS).tolist()
        
        print("🔍 전체 검색 파이프라인 테스트 시작")
        print("=" * 60)
        
        # 쿼리 처리 (최소 50개 문서 보장)
        results = search_generator.process_queries(test_queries, min_documents_per_query=50)
        
        # 결과 저장
        output_file = search_generator.save_results_to_json(results)
        
        # 결과 요약 출력
        print("\n📊 검색 결과 요약:")
        print(f"   총 처리된 질문: {len(results)}개")
        print(f"   성공한 질문: {len([r for r in results if 'error' not in r])}개")
        print(f"   총 찾은 문서: {sum(r.get('total_documents_found', 0) for r in results)}개")
        print(f"   결과 파일: {output_file}")
        
        # 첫 번째 결과 미리보기
        if results:
            first_result = results[0]
            print(f"\n📝 첫 번째 결과 미리보기:")
            print(f"   질문: {first_result['question'][:80]}...")
            print(f"   한국어 키워드: {first_result['keywords']['korean']}")
            print(f"   영어 키워드: {first_result['keywords']['english']}")
            print(f"   생성된 검색어: {first_result.get('search_queries', [])[:3]}...")
            print(f"   찾은 문서 수: {first_result['total_documents_found']}개")
            if first_result['documents']:
                print(f"   첫 번째 문서: {first_result['documents'][0]['title'][:60]}...")
        
        print("\n✅ 전체 검색 파이프라인 테스트 완료!")
        
        # 리소스 정리
        search_generator.close()
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        logging.error(f"테스트 실행 실패: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        test_full_search_pipeline()
    else:
        test_keyword_extraction()