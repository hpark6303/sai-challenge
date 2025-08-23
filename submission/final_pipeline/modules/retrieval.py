"""
검색 모듈
- 키워드 추출
- 문서 검색
- 재시도 로직
- 검색 결과 보충
"""

import time
import re
from typing import List, Dict, Tuple
from konlpy.tag import Okt
from .config import SEARCH_CONFIG, TEST_CONFIG, CRAG_CONFIG
from .prompting import PromptEngineer

class DocumentRetriever:
    """문서 검색기"""
    
    def __init__(self, api_client, gemini_client=None):
        """
        검색기 초기화
        
        Args:
            api_client: ScienceON API 클라이언트
            gemini_client: Gemini API 클라이언트 (CRAG용)
        """
        self.api_client = api_client
        self.gemini_client = gemini_client
        self.okt = Okt()
        self.prompt_engineer = PromptEngineer()
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        쿼리에서 키워드 추출 (고급 LLM 기반 추출 사용)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추출된 키워드 리스트
        """
        # LLM 기반 고급 키워드 추출 시도
        if self.gemini_client:
            try:
                advanced_keywords = self._extract_keywords_with_llm(query)
                if advanced_keywords:
                    print(f"   🧠 LLM 기반 고급 키워드 추출: {', '.join(advanced_keywords)}")
                    return advanced_keywords
            except Exception as e:
                print(f"   ⚠️  LLM 키워드 추출 실패, 기본 방식 사용: {e}")
        
        # 폴백: 기존 방식 사용
        return self._extract_keywords_basic(query)
    
    def _extract_keywords_with_llm(self, query: str) -> List[str]:
        """
        LLM을 사용한 고급 키워드 추출
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추출된 키워드 리스트
        """
        # 고급 키워드 생성 프롬프트 사용
        prompt = self.prompt_engineer.create_advanced_keyword_generation_prompt(query)
        
        try:
            response = self.gemini_client.generate_answer(prompt)
            
            # 응답에서 키워드 추출 (줄바꿈으로 구분된 키워드들)
            keywords = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # 불필요한 문자 제거
                    clean_keyword = line.replace('*', '').replace('-', '').strip()
                    if clean_keyword and len(clean_keyword) > 1:
                        keywords.append(clean_keyword)
            
            return keywords[:5]  # 최대 5개 키워드
            
        except Exception as e:
            print(f"   ⚠️  LLM 키워드 추출 중 오류: {e}")
            return []
    
    def _extract_keywords_basic(self, query: str) -> List[str]:
        """
        기본 키워드 추출 (기존 방식)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추출된 키워드 리스트
        """
        if self._is_korean(query):
            # 한국어: 명사 추출
            nouns = self.okt.nouns(query)
            keywords = [noun for noun in nouns if len(noun) > 1]
            return keywords[:5]
        else:
            # 영어: 개선된 불용어 제거
            stop_words = {
                # 기본 불용어
                'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                # 질문어 제거
                'how', 'what', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose',
                # 일반적인 동사 제거
                'can', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                # 기타 일반적인 단어들
                'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
                'we', 'you', 'he', 'she', 'his', 'her', 'our', 'your', 'my', 'me', 'i'
            }
            
            words = re.findall(r'\w+', query.lower())
            
            # 1단계: 불용어 제거
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # 2단계: 전문 용어 우선 선택
            technical_terms = []
            general_terms = []
            
            for word in filtered_words:
                # 전문 용어 판별 (길이가 길거나 특정 패턴)
                if (len(word) > 6 or 
                    word in ['neural', 'artificial', 'machine', 'learning', 'deep', 'network', 
                            'algorithm', 'model', 'system', 'method', 'approach', 'technique',
                            'sustainability', 'corporate', 'culture', 'development', 'management',
                            'analysis', 'research', 'study', 'framework', 'architecture']):
                    technical_terms.append(word)
                else:
                    general_terms.append(word)
            
            # 전문 용어를 먼저, 그 다음 일반 용어
            keywords = technical_terms + general_terms
            return keywords[:8]  # 더 많은 키워드 추출
    
    def extract_more_keywords(self, query: str, max_keywords: int = 10) -> List[str]:
        """
        더 많은 키워드를 추출하는 함수
        
        Args:
            query: 검색 쿼리
            max_keywords: 최대 키워드 수
            
        Returns:
            확장된 키워드 리스트
        """
        keywords = self.extract_keywords(query)
        
        # 기본 키워드가 부족하면 쿼리에서 추가 키워드 추출
        if len(keywords) < 3:
            words = query.split()
            for word in words:
                if len(word) > 2 and word not in keywords:
                    keywords.append(word)
                    if len(keywords) >= max_keywords:
                        break
        
        # 유사한 키워드 추가 (간단한 확장)
        expanded_keywords = keywords.copy()
        for keyword in keywords:
            if keyword.isascii():
                if keyword.endswith('s'):
                    expanded_keywords.append(keyword[:-1])  # 복수형 -> 단수형
                if keyword.endswith('ing'):
                    expanded_keywords.append(keyword[:-3])  # ~ing -> 기본형
                if keyword.endswith('ed'):
                    expanded_keywords.append(keyword[:-2])  # ~ed -> 기본형
        
        return list(set(expanded_keywords))[:max_keywords]
    
    def search_with_retry(self, query: str, max_retries: int = SEARCH_CONFIG['max_retries'], 
                         min_docs: int = SEARCH_CONFIG['min_docs']) -> List[Dict]:
        """
        재시도 로직이 포함된 문서 검색 (50개 문서 보장)
        
        Args:
            query: 검색 쿼리
            max_retries: 최대 재시도 횟수
            min_docs: 최소 필요 문서 수 (50개)
            
        Returns:
            검색된 문서 리스트 (최소 50개 보장)
        """
        all_docs = []
        keywords = self.extract_more_keywords(query)
        
        print(f"   🔍 추출된 키워드: {', '.join(keywords[:10])}")
        
        # 더 적극적인 키워드 확장
        expanded_keywords = self._expand_keywords_aggressively(query, keywords)
        
        for attempt in range(max_retries * 2):  # 더 많은 시도
            if len(all_docs) >= min_docs:
                break
                
            print(f"   - 검색 시도 {attempt + 1}/{max_retries * 2} (현재 {len(all_docs)}개 문서)")
            
            # 이번 시도에서 사용할 키워드들
            current_keywords = self._get_keywords_for_attempt_aggressive(expanded_keywords, attempt)
            
            # 디버그 모드에서 키워드 추출 과정 표시
            if TEST_CONFIG.get('debug_mode', False):
                print(f"   🔍 시도 {attempt + 1} 키워드: {', '.join(current_keywords)}")
                if attempt > 0:
                    print(f"   🔄 키워드 확장: {len(expanded_keywords)}개 → {len(current_keywords)}개")
            
            # 키워드별 검색 (더 많은 결과 요청)
            for keyword in current_keywords:
                try:
                    docs = self.api_client.search_articles(
                        keyword, 
                        row_count=50,  # 더 많은 결과 요청
                        fields=['title', 'abstract', 'CN']
                    )
                    all_docs.extend(docs)
                    time.sleep(SEARCH_CONFIG['api_delay'])
                    
                    # 충분한 문서가 있으면 중단
                    if len(all_docs) >= min_docs * 2:  # 여유분 확보
                        break
                        
                except Exception as e:
                    print(f"   ⚠️  키워드 '{keyword}' 검색 실패: {e}")
                    continue
            
            # 중복 제거
            all_docs = self._remove_duplicates(all_docs)
            
            if len(all_docs) >= min_docs:
                print(f"   ✅ 목표 문서 수 달성: {len(all_docs)}개")
                break
            else:
                print(f"   ⚠️  문서 수 부족: {len(all_docs)}개 (목표: {min_docs}개)")
        
        # 최종적으로 50개 미만이면 추가 검색
        if len(all_docs) < min_docs:
            print(f"   🚨 문서 수 부족! 추가 검색 시도...")
            additional_docs = self._emergency_search(query, min_docs - len(all_docs))
            all_docs.extend(additional_docs)
            all_docs = self._remove_duplicates(all_docs)
        
        print(f"   📊 최종 검색 결과: {len(all_docs)}개 문서")
        return all_docs[:min_docs]  # 정확히 50개 반환
    
    def _expand_keywords_aggressively(self, query: str, base_keywords: List[str]) -> List[str]:
        """
        더 적극적인 키워드 확장 (50개 문서 보장을 위해)
        """
        expanded = base_keywords.copy()
        
        # 1. 쿼리에서 단어 분리
        words = query.replace('?', '').replace('.', '').split()
        for word in words:
            if len(word) > 2 and word not in expanded:
                expanded.append(word)
        
        # 2. 관련 용어 추가
        related_terms = self._get_related_terms(query)
        expanded.extend(related_terms)
        
        # 3. 일반적인 학술 용어 추가
        academic_terms = ['연구', '분석', '방법', '결과', '시스템', '기술', '개발', '평가', '관리', '최적화']
        for term in academic_terms:
            if term not in expanded:
                expanded.append(term)
        
        return list(set(expanded))  # 중복 제거
    
    def _get_related_terms(self, query: str) -> List[str]:
        """
        쿼리와 관련된 용어들 추출
        """
        related = []
        
        # 간단한 규칙 기반 관련 용어 매핑
        term_mapping = {
            '인공지능': ['AI', '머신러닝', '딥러닝', '알고리즘'],
            '머신러닝': ['ML', '인공지능', '데이터', '모델'],
            '데이터': ['분석', '처리', '마이닝', '베이스'],
            '시스템': ['구현', '설계', '아키텍처', '플랫폼'],
            '보안': ['암호화', '인증', '권한', '프로토콜'],
            '네트워크': ['통신', '프로토콜', '라우팅', '보안'],
            '웹': ['인터넷', '브라우저', '서버', '클라이언트'],
            '모바일': ['스마트폰', '앱', '안드로이드', 'iOS'],
            '클라우드': ['서버', '가상화', '스케일링', '배포']
        }
        
        for key, terms in term_mapping.items():
            if key in query:
                related.extend(terms)
        
        return related
    
    def _get_keywords_for_attempt_aggressive(self, keywords: List[str], attempt: int) -> List[str]:
        """
        더 적극적인 시도별 키워드 선택
        """
        if attempt == 0:
            return keywords[:15]  # 첫 시도: 상위 15개
        elif attempt == 1:
            return keywords[15:30]  # 두 번째 시도: 다음 15개
        elif attempt == 2:
            return keywords[30:45]  # 세 번째 시도: 다음 15개
        else:
            # 추가 시도: 남은 모든 키워드
            start_idx = 45 + (attempt - 3) * 10
            return keywords[start_idx:start_idx + 15]
    
    def _emergency_search(self, query: str, needed_count: int) -> List[Dict]:
        """
        긴급 추가 검색 (50개 문서 보장을 위해)
        """
        print(f"   🚨 긴급 검색: {needed_count}개 문서 추가 필요")
        
        emergency_docs = []
        
        # 1. 일반적인 학술 키워드로 검색
        emergency_keywords = ['연구', '분석', '방법', '시스템', '기술', '개발']
        
        for keyword in emergency_keywords:
            if len(emergency_docs) >= needed_count:
                break
                
            try:
                docs = self.api_client.search_articles(
                    keyword,
                    row_count=20,
                    fields=['title', 'abstract', 'CN']
                )
                emergency_docs.extend(docs)
                time.sleep(SEARCH_CONFIG['api_delay'])
            except Exception as e:
                print(f"   ⚠️  긴급 검색 키워드 '{keyword}' 실패: {e}")
                continue
        
        # 2. 쿼리에서 단어 하나씩 검색
        words = query.replace('?', '').replace('.', '').split()
        for word in words:
            if len(word) > 2 and len(emergency_docs) < needed_count:
                try:
                    docs = self.api_client.search_articles(
                        word,
                        row_count=10,
                        fields=['title', 'abstract', 'CN']
                    )
                    emergency_docs.extend(docs)
                    time.sleep(SEARCH_CONFIG['api_delay'])
                except Exception as e:
                    continue
        
        print(f"   📊 긴급 검색 결과: {len(emergency_docs)}개 문서")
        return emergency_docs
    
    def _get_keywords_for_attempt(self, keywords: List[str], attempt: int) -> List[str]:
        """시도별 키워드 선택 (기존 메서드 유지)"""
        if attempt == 0:
            return keywords[:5]  # 첫 시도: 상위 5개 키워드
        elif attempt == 1:
            return keywords[5:] + keywords[:3]  # 두 번째: 나머지 + 상위 3개
        else:
            # 마지막 시도: 쿼리 자체를 키워드로 사용
            return [keywords[0][:20], keywords[0][20:40]] if len(keywords[0]) > 20 else [keywords[0]]
    
    def _remove_duplicates(self, documents: List[Dict]) -> List[Dict]:
        """중복 문서 제거 및 품질 필터링"""
        # 중복 제거
        unique_docs = list({doc['CN']: doc for doc in documents if 'CN' in doc}.values())
        
        # 품질 필터링
        filtered_docs = []
        for doc in unique_docs:
            title = doc.get('title', '').lower()
            abstract = doc.get('abstract', '')
            
            # 제외할 패턴들
            exclude_patterns = [
                # "How"로 시작하는 일반적인 질문 제목들
                title.startswith('how '),
                title.startswith('what '),
                title.startswith('why '),
                title.startswith('when '),
                title.startswith('where '),
                title.startswith('which '),
                title.startswith('who '),
                
                # 너무 짧은 제목
                len(title) < 10,
                
                # 초록이 없는 경우
                not abstract or len(abstract) < 20,
                
                # 특정 무관한 키워드가 제목에 포함된 경우
                any(word in title for word in ['economics', 'language', 'teaching', 'learning', 'education'])
            ]
            
            # 제외 패턴에 해당하지 않으면 포함
            if not any(exclude_patterns):
                filtered_docs.append(doc)
        
        print(f"   🔍 품질 필터링: {len(unique_docs)}개 → {len(filtered_docs)}개")
        return filtered_docs
    
    def _is_korean(self, text: str) -> bool:
        """한국어 텍스트 감지"""
        return bool(re.search('[가-힣]', text))
    
    def supplement_documents(self, vector_docs: List[Dict], original_docs: List[Dict], 
                           target_count: int = SEARCH_CONFIG['min_docs']) -> List[Dict]:
        """
        벡터 검색 결과가 부족할 때 원본 검색 결과로 보충
        
        Args:
            vector_docs: 벡터 검색 결과
            original_docs: 원본 검색 결과
            target_count: 목표 문서 수
            
        Returns:
            보충된 문서 리스트
        """
        if len(vector_docs) >= target_count:
            return vector_docs
        
        print(f"   ⚠️  벡터 검색 결과 부족: {len(vector_docs)}개 (목표: {target_count}개)")
        
        # 원본 검색 결과에서 추가 문서 가져오기
        remaining_docs = original_docs[len(vector_docs):target_count] if len(original_docs) > len(vector_docs) else []
        supplemented_docs = vector_docs + remaining_docs
        
        print(f"   ✅ 원본 검색 결과로 보충: {len(supplemented_docs)}개")
        
        return supplemented_docs[:target_count]

    def search_with_crag(self, query: str) -> List[Dict]:
        """
        CRAG 파이프라인을 사용한 문서 검색
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색된 문서 리스트
        """
        if not CRAG_CONFIG.get('enable_crag', False) or not self.gemini_client:
            print("   ⚠️  CRAG 비활성화 또는 Gemini 클라이언트 없음 - 일반 검색 사용")
            return self.search_with_retry(query)
        
        print("   🔄 CRAG 파이프라인 시작")
        
        # 1차 검색
        initial_docs = self.search_with_retry(query)
        print(f"   📚 1차 검색 결과: {len(initial_docs)}개 문서")
        
        # 품질 평가
        quality_score, issues = self._evaluate_search_quality(query, initial_docs)
        print(f"   📊 품질 평가 점수: {quality_score:.2f}/10")
        
        # 성공 여부 판단
        threshold = CRAG_CONFIG.get('quality_threshold', 0.7)
        if quality_score >= threshold * 10:  # 10점 만점 기준
            print("   ✅ 품질 기준 충족 - 1차 검색 결과 사용")
            return initial_docs
        
        # 실패 시 교정 검색
        print("   ⚠️  품질 기준 미달 - 교정 검색 시작")
        corrected_docs = self._corrective_search(query, initial_docs, issues)
        
        return corrected_docs
    
    def _evaluate_search_quality(self, query: str, documents: List[Dict]) -> Tuple[float, str]:
        """
        검색 결과 품질 평가
        
        Args:
            query: 원본 질문
            documents: 검색된 문서들
            
        Returns:
            (품질 점수, 문제점 설명) 튜플
        """
        if not documents:
            return 0.0, "검색 결과가 없습니다."
        
        # 샘플 문서 선택 (처음 3개)
        sample_docs = documents[:3]
        sample_text = "\n".join([
            f"제목: {doc.get('title', 'N/A')}\n초록: {doc.get('abstract', 'N/A')[:200]}..."
            for doc in sample_docs
        ])
        
        # 평가 프롬프트 생성
        prompt = CRAG_CONFIG['correction_prompt_template'].format(
            query=query,
            doc_count=len(documents),
            sample_docs=sample_text,
            relevance_score="",  # LLM이 채울 부분
            quality_score="",
            sufficiency_score="",
            total_score="",
            improvement_suggestions="",
            new_keywords=""
        )
        
        try:
            # Gemini로 품질 평가
            evaluation_text = self.gemini_client.generate_answer(prompt)
            
            # 점수 추출 (간단한 파싱)
            score_match = re.search(r'종합 점수:\s*(\d+(?:\.\d+)?)/10', evaluation_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                # 점수를 찾을 수 없으면 기본값
                score = 5.0
            
            # 문제점 추출
            issues_match = re.search(r'개선 제안:\s*(.*?)(?=\n\n|\n새로운 검색 키워드:|$)', 
                                   evaluation_text, re.DOTALL)
            issues = issues_match.group(1).strip() if issues_match else "평가 중 오류 발생"
            
            return score, issues
            
        except Exception as e:
            print(f"   ⚠️  품질 평가 실패: {e}")
            return 5.0, f"평가 중 오류: {str(e)}"
    
    def _corrective_search(self, query: str, original_docs: List[Dict], issues: str) -> List[Dict]:
        """
        교정 검색 수행
        
        Args:
            query: 원본 질문
            original_docs: 원본 검색 결과
            issues: 검색 결과 문제점
            
        Returns:
            교정된 검색 결과
        """
        print("   🔄 교정 검색 시작")
        
        max_attempts = CRAG_CONFIG.get('max_corrective_attempts', 2)
        
        for attempt in range(max_attempts):
            print(f"   - 교정 시도 {attempt + 1}/{max_attempts}")
            
            # 개선된 키워드 생성
            improved_keywords = self._generate_improved_keywords(query, issues)
            print(f"   🔍 개선된 키워드: {', '.join(improved_keywords)}")
            
            # 개선된 키워드로 재검색
            corrected_docs = []
            for keyword in improved_keywords:
                try:
                    docs = self.api_client.search_articles(
                        keyword, 
                        row_count=30,
                        fields=['title', 'abstract', 'CN']
                    )
                    corrected_docs.extend(docs)
                    time.sleep(SEARCH_CONFIG['api_delay'])
                except Exception as e:
                    print(f"   ⚠️  키워드 '{keyword}' 검색 실패: {e}")
                    continue
            
            # 중복 제거 및 품질 필터링
            corrected_docs = self._remove_duplicates(corrected_docs)
            
            # 품질 재평가
            if corrected_docs:
                quality_score, _ = self._evaluate_search_quality(query, corrected_docs)
                print(f"   📊 교정 후 품질 점수: {quality_score:.2f}/10")
                
                threshold = CRAG_CONFIG.get('quality_threshold', 0.7)
                if quality_score >= threshold * 10:
                    print("   ✅ 교정 검색 성공")
                    return corrected_docs
            
            print("   ⚠️  교정 검색 품질 미달 - 추가 시도")
        
        print("   🚨 모든 교정 시도 실패 - 원본 결과 반환")
        return original_docs
    
    def _generate_improved_keywords(self, query: str, issues: str) -> List[str]:
        """
        개선된 검색 키워드 생성
        
        Args:
            query: 원본 질문
            issues: 검색 결과 문제점
            
        Returns:
            개선된 키워드 리스트
        """
        # 웹 검색이 활성화되어 있으면 웹 검색 활용
        if CRAG_CONFIG.get('web_search_enabled', False):
            return self._generate_keywords_with_web_search(query, issues)
        else:
            return self._generate_keywords_with_llm(query, issues)
    
    def _generate_keywords_with_llm(self, query: str, issues: str) -> List[str]:
        """
        LLM을 사용한 키워드 개선 (고급 프롬프트 사용)
        """
        # 고급 키워드 생성 프롬프트 사용
        prompt = self.prompt_engineer.create_advanced_keyword_generation_prompt(query)
        
        # 문제점 정보 추가
        if issues:
            prompt += f"\n\n# 기존 검색 결과 문제점:\n{issues}\n\n위 문제점을 고려하여 더 정확한 키워드를 생성해주세요."
        
        try:
            response = self.gemini_client.generate_answer(prompt)
            
            # 응답에서 키워드 추출
            keywords = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    clean_keyword = line.replace('*', '').replace('-', '').strip()
                    if clean_keyword and len(clean_keyword) > 1:
                        keywords.append(clean_keyword)
            
            return keywords[:8]  # 최대 8개
            
        except Exception as e:
            print(f"   ⚠️  LLM 키워드 생성 실패: {e}")
            # 폴백: 기존 키워드 확장
            return self.extract_more_keywords(query, 8)
    
    def _generate_keywords_with_web_search(self, query: str, issues: str) -> List[str]:
        """
        웹 검색을 활용한 키워드 개선 (향후 구현)
        """
        # 현재는 LLM 방식 사용
        return self._generate_keywords_with_llm(query, issues)
