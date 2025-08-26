"""
프롬프트 엔지니어링 모듈
- 고품질 프롬프트 생성
- 언어별 프롬프트 최적화
- 답변 품질 보장
"""

import re
from typing import List, Dict
from .config import PROMPT_CONFIG, ANSWER_CONFIG

class PromptEngineer:
    """프롬프트 엔지니어"""
    
    def __init__(self):
        """프롬프트 엔지니어 초기화"""
        pass
    
    def create_final_prompt(self, query: str, context: str, language: str) -> str:
        """
        최종 답변 생성을 위한 프롬프트 (Chain of Thought + 예시 포함)
        
        Args:
            query: 사용자 질문
            context: 참고 문서 컨텍스트
            language: 언어 ('ko' 또는 'en')
            
        Returns:
            생성된 프롬프트
        """
        language_instruction = "한국어로" if language == "ko" else "영어로 (in English)"
        
        return f"""당신은 학술 연구 전문가입니다. 제공된 문서들의 내용을 바탕으로 질문에 대한 정확하고 전문적인 답변을 작성하세요.

## 📋 작업 과정 (Chain of Thought)

### 1단계: 질문 분석
- 질문의 핵심 주제와 요구사항을 파악하세요
- 어떤 종류의 정보가 필요한지 명확히 하세요

### 2단계: 문서 검토
- 제공된 문서들을 체계적으로 검토하세요
- 질문과 관련된 핵심 정보를 추출하세요
- 문서 간의 연결점과 차이점을 파악하세요

### 3단계: 정보 종합
- 추출한 정보를 논리적으로 연결하세요
- 일관성 있고 체계적인 구조로 정리하세요

### 4단계: 답변 구성
- 전문적이고 명확한 언어로 답변을 작성하세요
- 구체적인 예시와 데이터를 포함하세요

## 📚 참고 문서:
{context}

## ❓ 질문:
{query}

## 🎯 답변 작성 원칙:
1. **언어 일치**: 질문과 같은 언어로 답변하세요 ({language_instruction})
2. **전문성**: 해당 분야의 전문가 수준으로 답변하세요
3. **직접성**: 다음 메타 설명들을 절대 사용하지 마세요:
   - "제공된 문서를 바탕으로", "문서 분석을 통한"
   - "본 보고서는", "이 연구에서는"
   - "문서 1은", "문서 2는" 등의 참고 문헌 언급
   - "제시된 자료", "참고 문서" 등의 표현
4. **지식 전달**: 순수하게 지식과 정보만을 전달하세요
5. **구체성**: 추상적인 설명보다는 구체적인 내용을 제공하세요

## 📝 출력 형식:
{self._format_output_instructions()}

## 💡 좋은 답변 예시:

### 예시 질문: "빅데이터를 활용한 고객 행동 분석 방법의 주요 특징은 무엇인가요?"

### 예시 답변:
**제목: 빅데이터 기반 고객 행동 분석 방법론의 핵심 특징**

**서론:**
빅데이터 기술의 발전으로 고객 행동 분석이 혁신적으로 변화하고 있습니다. 이 분석 방법은 실시간 데이터 처리, 다차원 패턴 인식, 예측적 인사이트 도출을 통해 기업의 의사결정을 지원합니다.

**본론:**
1. **실시간 데이터 처리**: 스트리밍 기술을 활용한 즉시 분석으로 시의적절한 대응이 가능합니다.
2. **다차원 패턴 인식**: 구매 이력, 웹사이트 방문 패턴, 소셜 미디어 활동을 종합적으로 분석합니다.
3. **예측적 인사이트**: 머신러닝 알고리즘을 통해 고객의 미래 행동을 예측합니다.
4. **개인화 서비스**: 개별 고객의 선호도를 파악하여 맞춤형 서비스를 제공합니다.

**결론:**
빅데이터 기반 고객 행동 분석은 데이터의 양적 확장과 질적 향상을 통해 기업의 경쟁력을 크게 향상시킬 수 있습니다.

## ⚠️ 금지 표현 예시:
- ❌ "제공된 문서를 바탕으로..."
- ❌ "본 보고서는..."
- ❌ "문서 1에서는..."
- ❌ "이 연구에서는..."
- ❌ "제시된 자료에 따르면..."

---
## ✍️ 최종 답변:
"""
    
    def _format_output_instructions(self) -> str:
        """출력 형식 지침 포맷팅"""
        formatted = []
        for i, instruction in enumerate(PROMPT_CONFIG['output_format'], 1):
            formatted.append(f"{i}. **{instruction}**")
        return "\n".join(formatted)
    
    def create_simple_prompt(self, query: str, context: str) -> str:
        """
        간단한 프롬프트 (fallback용) - Chain of Thought 포함
        
        Args:
            query: 사용자 질문
            context: 참고 문서 컨텍스트
            
        Returns:
            간단한 프롬프트
        """
        return f"""당신은 학술 연구 전문가입니다. 다음 과정을 따라 질문에 답변하세요:

## 🔍 분석 과정:
1. 질문의 핵심 요구사항 파악
2. 제공된 문서에서 관련 정보 추출
3. 정보를 논리적으로 종합
4. 전문적이고 명확한 답변 작성

## 📚 참고 문서:
{context}

## ❓ 질문:
{query}

## ✍️ 답변 (최소 {ANSWER_CONFIG['min_answer_length']}자 이상):
- "제공된 문서를 바탕으로" 등의 메타 설명 제외
- 직접적이고 전문적인 내용으로 작성
- 구체적인 정보와 예시 포함
"""
    
    def create_quality_check_prompt(self, answer: str, query: str) -> str:
        """
        답변 품질 검증용 프롬프트
        
        Args:
            answer: 생성된 답변
            query: 원본 질문
            
        Returns:
            품질 검증 프롬프트
        """
        return f"""다음 답변이 질문에 적절하게 답변하고 있는지 평가해주세요.

질문: {query}

답변: {answer}

평가 기준:
1. 답변의 정확성 (0-10점)
2. 답변의 완성도 (0-10점)
3. 질문과의 관련성 (0-10점)

평가 결과:
"""

    def create_advanced_keyword_generation_prompt(self, question: str) -> str:
        """
        ScienceOn API에 최적화된 작은 단위 키워드를 직접 생성하는 프롬프트.
        """
        return f"""
# ROLE & GOAL
당신은 한국 학술 연구 데이터베이스 'ScienceOn'의 검색 성능을 극대화하는 전문가입니다. 
사용자의 질문을 분석하여 ScienceOn API에서 효과적으로 검색할 수 있는 **작은 단위의 키워드들**을 생성하세요.

# KEY REQUIREMENTS
1. **작은 단위 키워드**: 긴 문구 대신 1-3단어로 구성된 작은 키워드 생성
2. **즉시 검색 가능**: ScienceOn API에서 바로 검색할 수 있는 형태
3. **핵심 용어 우선**: 질문의 핵심 개념을 나타내는 전문 용어 위주
4. **중복 제거**: 비슷한 의미의 키워드는 하나로 통합

# EXAMPLES
❌ 잘못된 예시:
- "Big Data를 이용한 Warehouse Management System 모델"
- "Mechanical Turk 데이터로부터 TurKontrol의 POMDP 파라미터 학습"

✅ 올바른 예시:
- "Big Data"
- "Warehouse Management"
- "Mechanical Turk"
- "POMDP"
- "TurKontrol"

# PROCESS
1. 질문에서 핵심 개념 추출
2. 각 개념을 1-3단어로 분할
3. 검색 가능한 작은 키워드로 변환
4. 중복 제거 및 정리

# OUTPUT FORMAT
최대 8개의 작은 키워드를 줄바꿈으로 구분하여 출력하세요.
각 키워드는 1-3단어로 구성되어야 합니다.

# USER QUESTION:
{question}

# SEARCH KEYWORDS:
"""

    def create_english_prompt(self, query: str, context: str) -> str:
        """
        영어 질문을 위한 특화된 프롬프트 (Chain of Thought + 영어 예시)
        
        Args:
            query: 영어 질문
            context: 참고 문서 컨텍스트
            
        Returns:
            영어 특화 프롬프트
        """
        return f"""You are an academic research expert. Please provide a comprehensive and professional answer based on the provided documents.

## 📋 Analysis Process (Chain of Thought):

### Step 1: Question Analysis
- Identify the core topic and requirements of the question
- Determine what specific information is needed

### Step 2: Document Review
- Systematically review the provided documents
- Extract key information relevant to the question
- Identify connections and differences between documents

### Step 3: Information Synthesis
- Logically connect the extracted information
- Organize into a coherent and systematic structure

### Step 4: Answer Composition
- Write the answer in professional and clear language
- Include specific examples and data

## 📚 Reference Documents:
{context}

## ❓ Question:
{query}

## 🎯 Answer Writing Principles:
1. **Professional Tone**: Write as an expert in the field
2. **Directness**: Absolutely avoid these meta-explanations:
   - "Based on the provided documents"
   - "According to the research"
   - "The documents show that"
   - "This study indicates"
   - "The analysis reveals"
3. **Knowledge Transfer**: Focus purely on transferring knowledge and information
4. **Specificity**: Provide concrete details rather than abstract explanations
5. **Structure**: Organize with clear sections and logical flow

## 📝 Output Format:
1. **Title**: Concise and professional title
2. **Introduction**: Brief background and context
3. **Main Body**: Detailed analysis with specific points
4. **Conclusion**: Summary of key findings

## 💡 Good Answer Example:

### Example Question: "What are the key features of machine learning approaches in healthcare applications?"

### Example Answer:
**Title: Key Features of Machine Learning Approaches in Healthcare Applications**

**Introduction:**
Machine learning has revolutionized healthcare by enabling predictive diagnostics, personalized treatment plans, and automated medical image analysis. These approaches leverage vast amounts of patient data to improve clinical decision-making and patient outcomes.

**Main Body:**
1. **Predictive Diagnostics**: ML algorithms analyze patient data to predict disease risk and progression, enabling early intervention.
2. **Personalized Medicine**: Individual patient characteristics are used to tailor treatment plans and medication dosages.
3. **Medical Image Analysis**: Deep learning models provide accurate interpretation of X-rays, MRIs, and CT scans.
4. **Clinical Decision Support**: Real-time analysis of patient data assists healthcare providers in making informed decisions.

**Conclusion:**
Machine learning in healthcare represents a paradigm shift toward data-driven, personalized medicine that enhances both diagnostic accuracy and treatment effectiveness.

---
## ✍️ Your Answer:
"""

    def create_bilingual_keyword_prompt(self, question: str, target_language: str = "ko") -> str:
        """
        쌍방 언어 키워드 생성을 위한 프롬프트
        
        Args:
            question: 원본 질문
            target_language: 대상 언어 ('ko' 또는 'en')
            
        Returns:
            쌍방 언어 키워드 생성 프롬프트
        """
        language_instruction = "한국어" if target_language == "ko" else "영어"
        
        return f"""
당신은 다국어 학술 검색 전문가입니다. 주어진 질문을 분석하여 {language_instruction}로 된 고품질 검색 키워드를 생성해주세요.

# 질문 분석 및 키워드 생성 가이드라인:

1. **질문의 핵심 주제 파악**: 질문에서 가장 중요한 학술적 개념을 식별하세요.
2. **전문 용어 추출**: 해당 분야의 전문적인 용어와 개념을 추출하세요.
3. **동의어 및 관련어 확장**: 핵심 개념의 동의어, 유의어, 상위/하위 개념을 포함하세요.
4. **학술적 표현 사용**: 일반적인 단어보다는 학술 논문에서 사용되는 전문적인 표현을 선호하세요.

# 원본 질문:
{question}

# {language_instruction} 검색 키워드 (3-5개, 줄바꿈으로 구분):
"""
    
    def detect_language(self, text: str) -> str:
        """
        텍스트 언어 감지
        
        Args:
            text: 감지할 텍스트
            
        Returns:
            언어 코드 ('ko' 또는 'en')
        """
        korean_chars = len(re.findall('[가-힣]', text))
        total_chars = len(re.findall('[a-zA-Z가-힣]', text))
        
        if total_chars == 0:
            return 'en'  # 기본값
        
        korean_ratio = korean_chars / total_chars
        return 'ko' if korean_ratio > 0.3 else 'en'
    
    def enhance_context(self, context: str, query: str) -> str:
        """
        컨텍스트 강화
        
        Args:
            context: 원본 컨텍스트
            query: 사용자 질문
            
        Returns:
            강화된 컨텍스트
        """
        if not context:
            return context
        
        # 질문 키워드 강조
        query_keywords = re.findall(r'\w+', query.lower())
        
        enhanced_context = context
        for keyword in query_keywords[:5]:  # 상위 5개 키워드만
            if len(keyword) > 2:
                # 키워드가 포함된 문장 강조
                enhanced_context = enhanced_context.replace(
                    keyword, f"**{keyword}**"
                )
        
        return enhanced_context
    
    def create_fallback_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Fallback 답변 생성용 프롬프트
        
        Args:
            query: 사용자 질문
            documents: 참고 문서 리스트
            
        Returns:
            Fallback 프롬프트
        """
        if not documents:
            return f"질문 '{query}'에 대한 충분한 정보를 찾을 수 없습니다."
        
        # 문서에서 핵심 정보 추출
        titles = [doc.get('title', '') for doc in documents if doc.get('title')]
        
        language = self.detect_language(query)
        
        if language == "ko":
            prompt = f"제공된 문서들을 바탕으로 '{query}'에 대한 분석을 수행했습니다.\n\n"
            prompt += "주요 참고 문서:\n"
            for i, title in enumerate(titles[:3], 1):
                prompt += f"{i}. {title}\n"
            prompt += f"\n이 문서들은 질문과 관련된 유용한 정보를 제공합니다. 상세한 내용은 참고 문서를 확인하시기 바랍니다."
        else:
            prompt = f"Based on the provided documents, I have analyzed '{query}'.\n\n"
            prompt += "Key reference documents:\n"
            for i, title in enumerate(titles[:3], 1):
                prompt += f"{i}. {title}\n"
            prompt += f"\nThese documents provide useful information related to the question. Please refer to the documents for detailed content."
        
        return prompt
