"""
검색 모듈
- 키워드 추출
- 문서 검색
- 재시도 로직
- 검색 결과 보충
"""

import time
import re
from typing import List, Dict
from konlpy.tag import Okt
from .config import SEARCH_CONFIG, TEST_CONFIG

class DocumentRetriever:
    """문서 검색기"""
    
    def __init__(self, api_client):
        """
        검색기 초기화
        
        Args:
            api_client: ScienceON API 클라이언트
        """
        self.api_client = api_client
        self.okt = Okt()
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        쿼리에서 키워드 추출
        
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
