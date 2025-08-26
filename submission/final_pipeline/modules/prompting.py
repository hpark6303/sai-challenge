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
**언어 일치**: 질문과 같은 언어로 답변하세요 ({language_instruction})
**전문성**: 해당 분야의 전문가 수준으로 답변하세요
**직접성**: 다음 메타 설명들을 절대 사용하지 마세요:
   - "제공된 문서를 바탕으로", "문서 분석을 통한"
   - "본 보고서는", "이 연구에서는"
   - "문서 1은", "문서 2는" 등의 참고 문헌 언급
   - "제시된 자료", "참고 문서" 등의 표현
**지식 전달**: 순수하게 지식과 정보만을 전달하세요 구조화의 방법에 숫자를 활용하지 마세요. 
**구체성**: 추상적인 설명보다는 구체적인 내용을 제공하세요

## 📝 출력 형식:
{self._format_output_instructions()}

## 💡 좋은 답변 예시:

#예시 질문: 제조 품질 개선을 위해 제안된 식스 시그마 기반 Big Data 활용 방법의 주요 절차를 요약해 주시겠습니까?

#예시 답변:
##제목## 빅데이터를 활용한 식스 시그마 기반 제조 품질 개선 절차 요약 ##서론## 제조 기업은 전통적인 식스 시그마 프로젝트를 통해 체계적인 품질 개선을 추진해 왔으나, 최근 빅데이터 기술을 접목함으로써 문제점 탐색과 개선 효과 검증을 더욱 신속·정밀하게 수행할 수 있는 가능성이 커졌다. 본 연구에서는 식스 시그마의 DMAIC(Define-Measure-Analyze-Improve-Control) 단계별로 빅데이터 활용 방안을 제안한다. ##본론## 1. Define(정의) - 개선 목표 및 핵심 품질 이슈를 명확히 설정 - 빅데이터 플랫폼에 연계 가능한 공정·장비·검사 데이터 범위 지정 2. Measure(측정) - 센서, 생산관리시스템, 검사 장비 등에서 대량의 실시간 데이터를 수집·통합 - 데이터 정합성·이상치 검출을 위한 전처리 적용 3. Analyze(분석) - 통계적 기법 및 머신러닝 모델을 활용해 주요 결함 원인과 공정 변수 상관관계 파악 - 멀티변량 분석을 통해 숨겨진 패턴 및 잠재적 리스크 식별 4. Improve(개선) - 분석 결과를 바탕으로 공정 조건·검사 기준을 최적화 - 시뮬레이션 및 파일럿 실험을 통해 개선안의 실효성 검증 5. Control(관리) - 실시간 모니터링 대시보드와 이상 알림 시스템을 구축해 개선 결과 지속 관찰 - 제어 차트·경고 임계치 설정으로 재발 방지 및 표준화 유지 ##결론## 식스 시그마의 DMAIC 절차에 빅데이터 수집·분석·시각화 역량을 결합함으로써 품질 문제를 더욱 빠르고 정확하게 해결할 수 있으며, 지속적인 모니터링을 통해 제조 공정의 안정성과 경쟁력을 동시에 확보할 수 있다.

# 예시 질문:DBN 기반 딥 러닝을 이용한 기업부도 예측과 기존 SVM 방법 간의 성능 차이, 특히 부도기업 예측 민감도 향상 결과를 간단히 정리해 주실 수 있나요?

# 예시 답변: ##제목## DBN 기반 딥러닝과 SVM을 활용한 기업부도 예측 성능 비교 ##서론## 기업부도는 국가경제와 이해관계자들에게 심각한 손실을 초래하므로, 이를 정확히 예측하는 연구가 중요하다. 최근 이미지·음성·자연어 처리 분야에서 우수한 성능을 보인 Deep Belief Network(DBN)를 기업부도 예측에 도입하여 기존의 Support Vector Machine(SVM)과 비교 분석을 수행하였다. ##본론## - 연구 데이터 및 변수: 1999~2015년 코스닥·코스피 비금융업종 2,164개 기업(정상 1,669개, 부도 495개)과 한국은행 기업경영분석의 재무비율 변수 활용 - 모델 비교: DBN과 전통적 SVM을 동일 데이터로 학습·검증 - 주요 결과: 전반적 평가척도에서 DBN이 SVM보다 우수한 예측력을 보였으며, 특히 부도기업을 정확히 식별하는 민감도(sensitivity)가 시험 데이터 기준으로 5% 이상 높게 향상됨 ##결론## DBN 기반 딥러닝은 SVM 대비 부도기업 탐지 능력을 크게 개선하여, 기업부도 예측 분야에서 딥러닝 기법의 유용성을 확인시켜 주었다.

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
**Professional Tone**: Write as an expert in the field
**Directness**: Absolutely avoid these meta-explanations:
   - "Based on the provided documents"
   - "According to the research"
   - "The documents show that"
   - "This study indicates"
   - "The analysis reveals"
**Knowledge Transfer**: Focus purely on transferring knowledge and information
**Specificity**: Provide concrete details rather than abstract explanations
**Structure**: Organize with clear sections and logical flow

## 📝 Output Format:
**Title**: Concise and professional title
**Introduction**: Brief background and context
**Main Body**: Detailed analysis with specific points
**Conclusion**: Summary of key findings

## 💡 Good Answer Example:

#Example Question: "How would you concisely summarize the strategic landscape and major industry examples that characterize IT convergence developments in Korea?"
#Example Answer:##Strategic Landscape and Key Industry Cases of IT Convergence in Korea## ##Introduction## IT convergence in Korea has emerged as a core driver of national growth, combining information technology with traditional industries to foster new markets and enhance competitiveness. Government initiatives launched since 2008 have provided policy frameworks, R&D support and specialized convergence centers to accelerate cross–sector collaboration and standardization efforts. ##Main Body## Strategically, Korea benchmarks international convergence best practices while selectively focusing resources on promising fields such as u-IT, IT/OT and IT/BT fusion. In consumer electronics, LG and Samsung integrate sensors, network connectivity and multimedia platforms to deliver intelligent home appliances and smart displays. Heavy industry player POSCO employs IT to optimize steel production processes and develop smart factory solutions. The power sector’s Advanced Distribution Management System illustrates IT/OT convergence by merging SCADA, automation and global information-sharing functions for real-time grid control. Defense convergence models leverage commercial IT to improve weapon acquisition, command-and-control and logistics through dedicated defense IT convergence centers and new business-model frameworks. In agriculture and environment, smart-farm projects combine IoT sensors with climate control systems to promote low-carbon green growth, while healthcare and sports services use wearable u-IT devices and big-data analytics to enhance rehabilitation and performance monitoring. ##Conclusion## Korea’s IT convergence landscape is characterized by targeted government support, cross-industry standardization and leading examples in electronics, manufacturing, energy, defense and green industries. Sustained success will depend on ecosystem development, talent cultivation and continuous alignment of policy with emerging technological synergies.

#Example Question: "How can the rationale and structure of the free electronic textbook outlining the essential mathematics for understanding AI in a one- or two-semester undergraduate course be summarized?"
#Example Answer: ##Free Electronic Textbook on Essential Mathematics for AI## ##Introduction## As artificial intelligence permeates modern industries—from healthcare and robotics to smart homes and IoT—understanding its underlying mathematical principles has become indispensable for undergraduate students. To address this need, a research team developed a free electronic textbook titled “Fundamental Mathematics for AI,” designed to cover all core math concepts required for AI and machine learning within one or two semesters. ##Main Body## The textbook is organized into modular chapters that build progressively: it begins with vector and matrix operations fundamental to neural networks, then introduces probability theory and statistical inference for data modeling, followed by calculus and optimization techniques that underpin learning algorithms. Each module includes context-relevant examples, problem-solving exercises, and visualizations tailored to the local curriculum, ensuring practical comprehension. Accompanying online resources and interactive lectures support students from diverse majors, reinforcing theoretical material with hands-on applications in Python and MATLAB. The entire course framework—from learning objectives to assessment items—has been openly shared and successfully implemented at the undergraduate and graduate levels. ##Conclusion## By structuring essential topics into a cohesive, semester-based sequence and providing free, adaptable materials, this electronic textbook equips learners with the rigorous mathematical toolkit required for AI and facilitates broader access to high-quality instruction in rapidly evolving technological fields.

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
