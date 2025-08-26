"""
답변 생성 모듈
- Gemini API를 통한 답변 생성
- 품질 검증 및 재시도
- Fallback 답변 처리
"""

import time
from typing import List, Dict, Tuple
from .config import ANSWER_CONFIG
from .prompting import PromptEngineer

class AnswerGenerator:
    """답변 생성기"""
    
    def __init__(self, gemini_client):
        """
        답변 생성기 초기화
        
        Args:
            gemini_client: Gemini API 클라이언트
        """
        self.gemini_client = gemini_client
        self.prompt_engineer = PromptEngineer()
    
    def generate_answer(self, query: str, context: str, max_retries: int = ANSWER_CONFIG['max_retries']) -> str:
        """
        답변 생성 (재시도 로직 포함)
        
        Args:
            query: 사용자 질문
            context: 참고 문서 컨텍스트
            max_retries: 최대 재시도 횟수
            
        Returns:
            생성된 답변
        """
        # 언어 감지
        language = self.prompt_engineer.detect_language(query)
        
        # 컨텍스트 강화
        enhanced_context = self.prompt_engineer.enhance_context(context, query)
        
        # 언어에 따른 최적화된 프롬프트 생성
        if language == "en":
            # 영어 질문: 영어 특화 프롬프트 사용
            prompt = self.prompt_engineer.create_english_prompt(query, enhanced_context)
        else:
            # 한국어 질문: 한국어 특화 프롬프트 사용
            prompt = self.prompt_engineer.create_final_prompt(query, enhanced_context, language)
        
        # 답변 생성 (재시도 포함)
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.generate_answer(prompt)
                
                if response and self._validate_answer(response, query):
                    return response.strip()
                else:
                    print(f"   ⚠️  시도 {attempt + 1}: 답변 품질 부족")
                    
            except Exception as e:
                print(f"   ⚠️  시도 {attempt + 1}: API 호출 실패 - {str(e)[:50]}...")
            
            # 재시도 간 대기
            if attempt < max_retries - 1:
                time.sleep(1)
        
        # 모든 시도 실패 시 fallback 답변
        return self._generate_fallback_answer(query)
    
    def _validate_answer(self, answer: str, query: str) -> bool:
        """
        답변 품질 검증 (강화된 버전)
        
        Args:
            answer: 생성된 답변
            query: 원본 질문
            
        Returns:
            품질 검증 결과
        """
        if not answer or not answer.strip():
            return False
        
        # 최소 길이 검증
        if len(answer.strip()) < ANSWER_CONFIG['min_answer_length']:
            return False
        
        # 기본적인 품질 검증
        if answer.lower() in ['답변을 생성할 수 없습니다', 'error', 'failed', 'cannot generate']:
            return False
        
        # 메타 설명 검증 (제거된 메타 설명이 다시 나타나는지 확인)
        meta_phrases = [
            '제공된 문서를 바탕으로', '문서 분석을 통한', '참고문헌을 통해',
            'based on the provided documents', 'document analysis shows', 'according to the references'
        ]
        
        answer_lower = answer.lower()
        for phrase in meta_phrases:
            if phrase in answer_lower:
                print(f"   ⚠️  메타 설명 감지: {phrase}")
                return False
        
        # 구조 검증 (제목, 본문, 결론 포함 여부)
        has_title = any(keyword in answer for keyword in ['제목:', 'Title:', '**제목**', '**Title**'])
        has_body = any(keyword in answer for keyword in ['본론:', 'Main Body:', '**본론**', '**Main Body**'])
        has_conclusion = any(keyword in answer for keyword in ['결론:', 'Conclusion:', '**결론**', '**Conclusion**'])
        
        # 최소한 제목과 본문은 있어야 함
        if not (has_title or has_body):
            return False
        
        return True
    
    def _generate_fallback_answer(self, query: str) -> str:
        """
        Fallback 답변 생성
        
        Args:
            query: 사용자 질문
            
        Returns:
            Fallback 답변
        """
        language = self.prompt_engineer.detect_language(query)
        
        if language == "ko":
            return f"질문 '{query}'에 대한 답변을 생성하는 중 오류가 발생했습니다. 제공된 참고 문서를 확인해주시기 바랍니다."
        else:
            return f"An error occurred while generating an answer for the question '{query}'. Please check the provided reference documents."
    
    def generate_quality_answer(self, query: str, documents: List[Dict], 
                              max_retries: int = ANSWER_CONFIG['max_retries']) -> str:
        """
        품질이 보장된 답변 생성
        
        Args:
            query: 사용자 질문
            documents: 참고 문서 리스트
            max_retries: 최대 재시도 횟수
            
        Returns:
            품질 보장된 답변
        """
        if not documents:
            return self._generate_fallback_answer(query)
        
        # 컨텍스트 생성
        context = self._create_context_from_documents(documents)
        
        # 답변 생성
        answer = self.generate_answer(query, context, max_retries)
        
        # 품질 재검증
        if not self._validate_answer(answer, query):
            # 간단한 프롬프트로 재시도
            simple_prompt = self.prompt_engineer.create_simple_prompt(query, context)
            try:
                answer = self.gemini_client.generate_answer(simple_prompt)
                if answer:
                    answer = answer.strip()
            except:
                pass
        
        return answer if self._validate_answer(answer, query) else self._generate_fallback_answer(query)
    
    def _create_context_from_documents(self, documents: List[Dict]) -> str:
        """
        문서들로부터 확장된 컨텍스트 생성 (대회 핵심 요구사항)
        
        Args:
            documents: 문서 리스트
            
        Returns:
            확장된 컨텍스트
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents[:ANSWER_CONFIG['max_context_docs']]):
            # 문서 확장
            expanded_content = self._expand_document_content(doc)
            
            context = f"[문서 {i+1}]\n"
            context += f"제목: {doc.get('title', '')}\n"
            context += f"확장된 내용: {expanded_content}\n"
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def _expand_document_content(self, document: Dict) -> str:
        """
        문서 내용 확장 (대회 핵심 요구사항)
        
        Args:
            document: 원본 문서
            
        Returns:
            확장된 문서 내용
        """
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        
        # 1. 기본 내용
        expanded = abstract
        
        # 2. 제목에서 핵심 개념 추출 및 설명 추가
        title_concepts = self._extract_concepts_from_title(title)
        if title_concepts:
            expanded += f"\n\n핵심 개념: {', '.join(title_concepts)}"
        
        # 3. 방법론/기술 추출 및 설명
        methodologies = self._extract_methodologies(abstract)
        if methodologies:
            expanded += f"\n\n주요 방법론: {', '.join(methodologies)}"
        
        # 4. 결과/성과 추출
        results = self._extract_results(abstract)
        if results:
            expanded += f"\n\n주요 결과: {', '.join(results)}"
        
        # 5. 응용 분야 추출
        applications = self._extract_applications(abstract)
        if applications:
            expanded += f"\n\n응용 분야: {', '.join(applications)}"
        
        return expanded
    
    def _extract_concepts_from_title(self, title: str) -> List[str]:
        """제목에서 핵심 개념 추출"""
        concepts = []
        
        # 전문 용어 패턴 매칭
        technical_terms = [
            'neural network', 'machine learning', 'deep learning', 'artificial intelligence',
            'algorithm', 'framework', 'methodology', 'approach', 'technique',
            'sustainability', 'corporate culture', 'management', 'strategy',
            'mathematics', 'engineering', 'medical', 'clinical'
        ]
        
        title_lower = title.lower()
        for term in technical_terms:
            if term in title_lower:
                concepts.append(term)
        
        return concepts[:3]  # 상위 3개만
    
    def _extract_methodologies(self, abstract: str) -> List[str]:
        """초록에서 방법론 추출"""
        methodologies = []
        
        # 방법론 관련 키워드
        method_keywords = [
            'method', 'approach', 'technique', 'algorithm', 'framework',
            'model', 'system', 'procedure', 'strategy', 'methodology'
        ]
        
        sentences = abstract.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in method_keywords:
                if keyword in sentence_lower:
                    # 해당 문장에서 핵심 부분 추출
                    start = max(0, sentence_lower.find(keyword) - 50)
                    end = min(len(sentence), sentence_lower.find(keyword) + 100)
                    methodologies.append(sentence[start:end].strip())
                    break
        
        return methodologies[:2]  # 상위 2개만
    
    def _extract_results(self, abstract: str) -> List[str]:
        """초록에서 결과 추출"""
        results = []
        
        # 결과 관련 키워드
        result_keywords = [
            'result', 'outcome', 'performance', 'accuracy', 'efficiency',
            'improvement', 'enhancement', 'effectiveness', 'success'
        ]
        
        sentences = abstract.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in result_keywords:
                if keyword in sentence_lower:
                    start = max(0, sentence_lower.find(keyword) - 30)
                    end = min(len(sentence), sentence_lower.find(keyword) + 80)
                    results.append(sentence[start:end].strip())
                    break
        
        return results[:2]  # 상위 2개만
    
    def _extract_applications(self, abstract: str) -> List[str]:
        """초록에서 응용 분야 추출"""
        applications = []
        
        # 응용 분야 키워드
        application_keywords = [
            'application', 'use', 'implement', 'deploy', 'apply',
            'industry', 'field', 'domain', 'sector', 'area'
        ]
        
        sentences = abstract.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in application_keywords:
                if keyword in sentence_lower:
                    start = max(0, sentence_lower.find(keyword) - 40)
                    end = min(len(sentence), sentence_lower.find(keyword) + 60)
                    applications.append(sentence[start:end].strip())
                    break
        
        return applications[:2]  # 상위 2개만
    
    def batch_generate_answers(self, questions: List[Tuple[int, str]], 
                             documents_list: List[List[Dict]]) -> List[str]:
        """
        배치 답변 생성
        
        Args:
            questions: (질문 ID, 질문) 튜플 리스트
            documents_list: 각 질문별 문서 리스트
            
        Returns:
            생성된 답변 리스트
        """
        answers = []
        
        for (question_id, query), documents in zip(questions, documents_list):
            print(f"   🔍 질문 {question_id+1} 답변 생성 중...")
            
            answer = self.generate_quality_answer(query, documents)
            answers.append(answer)
            
            # API 호출 간격 조절
            time.sleep(0.5)
        
        return answers
