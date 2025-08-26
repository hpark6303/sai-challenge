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
        기본 키워드 추출 (개선된 방식)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추출된 키워드 리스트
        """
        if self._is_korean(query):
            # 한국어: 명사 추출 + 전문 용어 보존
            nouns = self.okt.nouns(query)
            keywords = [noun for noun in nouns if len(noun) > 1]
            
            # 전문 용어 및 약어 보존
            special_terms = self._extract_special_terms(query)
            keywords.extend(special_terms)
            
            # 중복 제거 및 정렬
            keywords = list(set(keywords))
            keywords.sort(key=len, reverse=True)  # 긴 키워드 우선
            
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
        단순화된 검색 전략 (빠른 검색)
        
        Args:
            query: 검색 쿼리
            max_retries: 최대 재시도 횟수
            min_docs: 최소 필요 문서 수 (50개)
            
        Returns:
            검색된 문서 리스트 (최소 50개 보장)
        """
        all_docs = []
        
        # 1단계: LLM 기반 작은 키워드로 직접 검색
        keywords = self.extract_keywords(query)
        print(f"   🔍 검색 키워드: {', '.join(keywords)}")
        
        for keyword in keywords:
            try:
                docs = self.api_client.search_articles(keyword, row_count=25, fields=['title', 'abstract', 'CN'])
                all_docs.extend(docs)
                print(f"   ✅ 키워드 '{keyword}'로 {len(docs)}개 문서 검색")
                time.sleep(SEARCH_CONFIG['api_delay'])
                
                if len(all_docs) >= min_docs:
                    break
            except Exception as e:
                print(f"   ⚠️ 키워드 '{keyword}' 검색 실패: {e}")
        
        # 2단계: 부족하면 기본 키워드로 보충
        if len(all_docs) < min_docs:
            print(f"   🔄 문서 부족 ({len(all_docs)}개), 기본 키워드로 보충")
            basic_keywords = self._extract_basic_keywords(query)
            
            for keyword in basic_keywords:
                if keyword not in keywords:
                    try:
                        docs = self.api_client.search_articles(keyword, row_count=15, fields=['title', 'abstract', 'CN'])
                        all_docs.extend(docs)
                        print(f"   ✅ 기본 키워드 '{keyword}'로 {len(docs)}개 문서 검색")
                        time.sleep(SEARCH_CONFIG['api_delay'])
                    except Exception as e:
                        print(f"   ⚠️ 기본 키워드 '{keyword}' 검색 실패: {e}")
                    
                    if len(all_docs) >= min_docs:
                        break
        
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
        단순화된 CRAG 파이프라인 (빠른 검색)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            검색된 문서 리스트
        """
        print("   🔄 CRAG 파이프라인 시작")
        
        # 직접 검색 (교정 과정 제거)
        documents = self.search_with_retry(query)
        print(f"   📚 검색 결과: {len(documents)}개 문서")
        
        return documents
    
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
            
            # 개선된 키워드 생성 (기존 키워드와 중복 제거)
            improved_keywords = self._generate_improved_keywords(query, issues)
            
            # 기존 키워드와 중복되지 않는 키워드만 필터링
            original_keywords = self.extract_keywords(query)
            unique_improved_keywords = [kw for kw in improved_keywords if kw not in original_keywords]
            
            if not unique_improved_keywords:
                print("   ⚠️  교정 키워드가 기존 키워드와 중복됨 - 대안 키워드 생성")
                unique_improved_keywords = self._generate_alternative_keywords(query, original_keywords)
            
            split_improved_keywords = self._split_keywords_by_length(unique_improved_keywords)
            print(f"   🔍 개선된 키워드: {', '.join(improved_keywords)}")
            print(f"   🔧 중복 제거된 키워드: {', '.join(unique_improved_keywords)}")
            print(f"   🔧 분할된 개선 키워드: {', '.join(split_improved_keywords)}")
            
            # 개선된 키워드로 재검색
            corrected_docs = []
            for keyword in split_improved_keywords:
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
        LLM을 사용한 키워드 개선 (교정 검색용 특화 프롬프트)
        """
        # 교정 검색용 특화 프롬프트
        prompt = f"""
# 교정 검색 키워드 생성

## 원본 질문:
{query}

## 기존 검색 결과 문제점:
{issues}

## 교정 검색 전략:
기존 키워드와는 다른 접근 방식으로 키워드를 생성해주세요:

1. **동의어/유사어 활용**: 기존 키워드의 동의어나 유사어 사용
2. **상위/하위 개념**: 더 넓은 범위나 더 구체적인 개념으로 확장
3. **관련 기술/방법론**: 같은 분야의 다른 기술이나 방법론
4. **영문/한글 변환**: 영문 키워드를 한글로, 한글 키워드를 영문으로
5. **약어/전체명**: 약어가 있다면 전체명으로, 전체명이 있다면 약어로

## 예시:
- "Machine Learning" → "딥러닝", "인공지능", "AI", "Neural Network"
- "크라우드소싱" → "Crowdsourcing", "Human Computation", "Distributed Computing"
- "POMDP" → "Partially Observable Markov Decision Process", "강화학습", "Reinforcement Learning"

## 요구사항:
- 기존 키워드와 중복되지 않는 새로운 키워드 생성
- 최대 5개의 핵심 키워드만 생성
- 줄바꿈으로 구분하여 출력

# 교정 키워드:
"""
        
        try:
            response = self.gemini_client.generate_answer(prompt)
            
            # 응답에서 키워드 추출
            keywords = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-') and not line.startswith('##'):
                    clean_keyword = line.replace('*', '').replace('-', '').strip()
                    if clean_keyword and len(clean_keyword) > 1:
                        keywords.append(clean_keyword)
            
            return keywords[:5]  # 최대 5개
            
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
    
    def _extract_special_terms(self, query: str) -> List[str]:
        """
        일반화된 전문 용어 및 약어 추출
        
        Args:
            query: 검색 쿼리
            
        Returns:
            추출된 전문 용어 리스트
        """
        special_terms = []
        
        # 1. 대문자 약어 추출 (DTG, SBiFEM, CNN, LSTM, SVM 등)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        special_terms.extend(acronyms)
        
        # 2. 복합 명사구 추출 (2-4단어 조합)
        words = query.split()
        for i in range(len(words) - 1):
            # 2단어 조합
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5 and not any(word in ['the', 'and', 'or', 'in', 'of', 'to', 'for'] for word in bigram.split()):
                special_terms.append(bigram)
            
            # 3단어 조합 (가능한 경우)
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram) > 8 and not any(word in ['the', 'and', 'or', 'in', 'of', 'to', 'for'] for word in trigram.split()):
                    special_terms.append(trigram)
        
        # 3. 특수 패턴 추출 (예: "X 기반 Y", "X를 활용한 Y" 등)
        patterns = [
            r'(\w+)\s+기반\s+(\w+)',
            r'(\w+)\s+활용\s+(\w+)',
            r'(\w+)\s+모델\s+(\w+)',
            r'(\w+)\s+시스템\s+(\w+)',
            r'(\w+)\s+분석\s+(\w+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                if isinstance(match, tuple):
                    combined = ' '.join(match)
                    if len(combined) > 3:
                        special_terms.append(combined)
                else:
                    special_terms.append(match)
        
        return list(set(special_terms))  # 중복 제거
    
    def _split_long_keywords(self, keywords: List[str]) -> List[str]:
        """
        긴 키워드를 쪼개서 검색 가능한 단위로 분할 (일반화된 방식)
        
        Args:
            keywords: 원본 키워드 리스트
            
        Returns:
            분할된 키워드 리스트
        """
        split_keywords = []
        
        for keyword in keywords:
            # 이미 짧은 키워드는 그대로 유지
            if len(keyword) <= 8:
                split_keywords.append(keyword)
                continue
            
            # 긴 키워드 분할
            if ' ' in keyword:
                # 공백이 있는 경우 단어별로 분할
                words = keyword.split()
                split_keywords.extend(words)
                
                # 2-3단어 조합도 추가 (검색 범위 확장)
                if len(words) >= 2:
                    for i in range(len(words) - 1):
                        split_keywords.append(f"{words[i]} {words[i+1]}")
                    
                    # 3단어 조합도 추가 (가능한 경우)
                    if len(words) >= 3:
                        for i in range(len(words) - 2):
                            split_keywords.append(f"{words[i]} {words[i+1]} {words[i+2]}")
            else:
                # 단일 긴 단어인 경우
                if len(keyword) > 12:
                    # 일반적인 영어 단어 분할 규칙 적용
                    # 1. camelCase 분할
                    if re.match(r'^[a-z]+[A-Z]', keyword):
                        # camelCase를 단어로 분할
                        words = re.findall(r'[A-Z]?[a-z]+', keyword)
                        split_keywords.extend(words)
                        # 원본도 유지
                        split_keywords.append(keyword)
                    # 2. snake_case 분할
                    elif '_' in keyword:
                        words = keyword.split('_')
                        split_keywords.extend(words)
                        # 원본도 유지
                        split_keywords.append(keyword)
                    # 3. 일반적인 긴 단어는 그대로 유지
                    else:
                        split_keywords.append(keyword)
                else:
                    split_keywords.append(keyword)
        
        # 중복 제거 및 정렬
        return list(set(split_keywords))
    
    def _split_keywords_by_length(self, keywords: List[str]) -> List[str]:
        """
        키워드를 스마트하게 분할하여 검색 범위 확장 (개선된 방식)
        
        Args:
            keywords: 원본 키워드 리스트
            
        Returns:
            분할된 키워드 리스트
        """
        split_keywords = []
        
        # 불용어 정의 (분할하지 않을 일반적인 단어들)
        stop_words = {
            'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
            '최적화', '분석', '연구', '방법', '기법', '시스템', '모델', '데이터', '정보'
        }
        
        for keyword in keywords:
            # 원본 키워드 추가
            split_keywords.append(keyword)
            
            # 공백이 있는 복합 키워드 처리
            if ' ' in keyword:
                words = keyword.split()
                
                # 의미있는 단어만 필터링
                meaningful_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
                
                # 개별 의미있는 단어 추가
                split_keywords.extend(meaningful_words)
                
                # 2-3단어 조합만 추가 (너무 많은 조합 방지)
                if len(meaningful_words) >= 2:
                    # 2단어 조합
                    for i in range(len(meaningful_words) - 1):
                        phrase = f"{meaningful_words[i]} {meaningful_words[i+1]}"
                        split_keywords.append(phrase)
                    
                    # 3단어 조합 (가능한 경우)
                    if len(meaningful_words) >= 3:
                        for i in range(len(meaningful_words) - 2):
                            phrase = f"{meaningful_words[i]} {meaningful_words[i+1]} {meaningful_words[i+2]}"
                            split_keywords.append(phrase)
            
            # 단일 긴 단어 처리
            elif len(keyword) > 8:
                # camelCase 분할
                if re.match(r'^[a-z]+[A-Z]', keyword):
                    words = re.findall(r'[A-Z]?[a-z]+', keyword)
                    meaningful_words = [word for word in words if word.lower() not in stop_words]
                    split_keywords.extend(meaningful_words)
                
                # snake_case 분할
                elif '_' in keyword:
                    words = keyword.split('_')
                    meaningful_words = [word for word in words if word.lower() not in stop_words]
                    split_keywords.extend(meaningful_words)
        
        # 중복 제거 및 길이별 정렬 (긴 키워드 우선)
        unique_keywords = list(set(split_keywords))
        unique_keywords.sort(key=len, reverse=True)
        
        return unique_keywords
    
    def _generate_alternative_keywords(self, query: str, existing_keywords: List[str]) -> List[str]:
        """
        기존 키워드와 다른 대안 키워드 생성
        
        Args:
            query: 원본 질문
            existing_keywords: 기존 키워드 리스트
            
        Returns:
            대안 키워드 리스트
        """
        alternative_keywords = []
        
        # 동의어 사전 (간단한 버전)
        synonyms = {
            'machine learning': ['딥러닝', '인공지능', 'AI', 'Neural Network', '강화학습'],
            'deep learning': ['머신러닝', '인공지능', 'AI', 'Neural Network', '딥러닝'],
            'artificial intelligence': ['AI', '머신러닝', '딥러닝', '인공지능'],
            'crowdsourcing': ['크라우드소싱', 'Human Computation', 'Distributed Computing'],
            '크라우드소싱': ['Crowdsourcing', 'Human Computation', 'Distributed Computing'],
            'warehouse': ['창고', '물류', 'Logistics', 'Supply Chain'],
            'management': ['관리', '운영', 'Administration', 'Control'],
            'system': ['시스템', '체계', 'Framework', 'Architecture'],
            'data': ['데이터', '정보', 'Information', 'Dataset'],
            'analysis': ['분석', 'Analytics', 'Processing', 'Evaluation'],
            'model': ['모델', 'Model', 'Algorithm', 'Method'],
            'optimization': ['최적화', 'Optimization', 'Improvement', 'Enhancement'],
            '최적화': ['Optimization', 'Improvement', 'Enhancement', 'Efficiency'],
            'pomdp': ['Partially Observable Markov Decision Process', '강화학습', 'Reinforcement Learning'],
            'reinforcement learning': ['강화학습', 'POMDP', 'Q-Learning', 'Policy Gradient'],
            '강화학습': ['Reinforcement Learning', 'POMDP', 'Q-Learning', 'Policy Gradient']
        }
        
        # 기존 키워드에 대한 동의어 찾기
        for keyword in existing_keywords:
            keyword_lower = keyword.lower()
            for key, values in synonyms.items():
                if key in keyword_lower or keyword_lower in key:
                    for synonym in values:
                        if synonym not in existing_keywords and synonym not in alternative_keywords:
                            alternative_keywords.append(synonym)
        
        # 도메인별 일반 키워드 추가
        if any(term in query.lower() for term in ['system', 'model', 'algorithm']):
            alternative_keywords.extend(['Framework', 'Architecture', 'Methodology'])
        if any(term in query.lower() for term in ['data', 'analysis', 'processing']):
            alternative_keywords.extend(['Analytics', 'Processing', 'Evaluation'])
        if any(term in query.lower() for term in ['learning', 'training']):
            alternative_keywords.extend(['Training', 'Education', 'Development'])
        
        return alternative_keywords[:5]  # 최대 5개
    
    def _extract_basic_keywords(self, query: str) -> List[str]:
        """
        기본 키워드 추출 (빠른 보충용)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            기본 키워드 리스트
        """
        basic_keywords = []
        
        # 1. 명사 추출
        if self._is_korean(query):
            nouns = self.okt.nouns(query)
            basic_keywords.extend([noun for noun in nouns if len(noun) > 1])
        else:
            # 영어: 기본 단어 추출
            words = re.findall(r'\b\w+\b', query.lower())
            basic_keywords.extend([word for word in words if len(word) > 2])
        
        # 2. 약어 추출
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        basic_keywords.extend(acronyms)
        
        # 3. 중복 제거 및 정렬
        basic_keywords = list(set(basic_keywords))
        basic_keywords.sort(key=len, reverse=True)
        
        return basic_keywords[:5]  # 최대 5개
    
    def _extract_general_keywords(self, query: str) -> List[str]:
        """
        일반적인 키워드 추출 (검색 실패 시 폴백용)
        
        Args:
            query: 검색 쿼리
            
        Returns:
            일반적인 키워드 리스트
        """
        general_keywords = []
        
        # 1. 기본 명사 추출
        if self._is_korean(query):
            nouns = self.okt.nouns(query)
            general_keywords.extend([noun for noun in nouns if len(noun) > 1])
        else:
            # 영어: 기본 단어 추출
            words = re.findall(r'\b\w+\b', query.lower())
            general_keywords.extend([word for word in words if len(word) > 2])
        
        # 2. 도메인별 일반 키워드 추가
        domain_keywords = self._get_domain_keywords(query)
        general_keywords.extend(domain_keywords)
        
        # 3. 중복 제거 및 정렬
        general_keywords = list(set(general_keywords))
        general_keywords.sort(key=len, reverse=True)
        
        return general_keywords[:10]
    
    def _get_domain_keywords(self, query: str) -> List[str]:
        """
        질문 도메인에 따른 일반 키워드 추출
        
        Args:
            query: 검색 쿼리
            
        Returns:
            도메인별 일반 키워드
        """
        query_lower = query.lower()
        domain_keywords = []
        
        # 기술/공학 도메인
        if any(term in query_lower for term in ['system', 'model', 'algorithm', 'method', 'approach']):
            domain_keywords.extend(['system', 'model', 'algorithm', 'method', 'approach', 'technique'])
        
        # 데이터/분석 도메인
        if any(term in query_lower for term in ['data', 'analysis', 'processing', 'mining', 'big data']):
            domain_keywords.extend(['data', 'analysis', 'processing', 'mining', 'big data', 'analytics'])
        
        # AI/ML 도메인
        if any(term in query_lower for term in ['ai', 'machine learning', 'deep learning', 'neural']):
            domain_keywords.extend(['artificial intelligence', 'machine learning', 'deep learning', 'neural network'])
        
        # 의료/생명 도메인
        if any(term in query_lower for term in ['medical', 'health', 'disease', 'diagnosis', 'treatment']):
            domain_keywords.extend(['medical', 'health', 'disease', 'diagnosis', 'treatment', 'clinical'])
        
        # 비즈니스/경영 도메인
        if any(term in query_lower for term in ['management', 'business', 'corporate', 'organization']):
            domain_keywords.extend(['management', 'business', 'corporate', 'organization', 'strategy'])
        
        return domain_keywords
