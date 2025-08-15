#!/usr/bin/env python3
"""
Kaggle 제출용 RAG 파이프라인 v9.0
- test.csv 형식에 맞춰 출력
- 한국어/영어 번역 처리
- 5개 논문 검색 (title + abstract + source 형식)
- 배치 처리로 최적화
"""

import json
from pathlib import Path
import sys
import time
import pandas as pd
import re
from tqdm import tqdm
from typing import List, Dict
import numpy as np
from collections import Counter

# --- 필요한 라이브러리 임포트 ---
try:
    from konlpy.tag import Okt
    from scienceon_api_example import ScienceONAPIClient
    from gemini_client import GeminiClient
except ImportError as e:
    print(f"🚨 [오류] 필수 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치해주세요: pip install tqdm konlpy")
    sys.exit(1)

def validate_credentials(path: Path) -> dict:
    """API 인증 정보 검증"""
    credentials = {}
    if not path.exists():
        print(f"🚨 설정 파일을 찾을 수 없습니다! (경로: {path})")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            credentials = json.load(f)
    except json.JSONDecodeError:
        print(f"🚨 설정 파일의 형식이 잘못되었습니다! (경로: {path})")
        sys.exit(1)
    
    required_keys = ['auth_key', 'client_id', 'mac_address']
    missing_keys = [key for key in required_keys if key not in credentials or not credentials[key]]
    if missing_keys:
        print(f"🚨 설정 파일에 필수 정보가 누락되었습니다! (누락된 정보: {', '.join(missing_keys)})")
        sys.exit(1)
    
    if len(credentials['auth_key']) != 32:
        print(f"🚨 인증키(auth_key)의 길이가 32자가 아닙니다! (현재 길이: {len(credentials['auth_key'])}자)")
        sys.exit(1)
    
    print("✅ [성공] API 인증 정보 파일이 유효합니다.")
    return credentials

def is_korean(text: str) -> bool:
    """한국어 텍스트 감지"""
    return bool(re.search('[가-힣]', text))

def extract_english_keywords(text: str) -> List[str]:
    """영어 키워드 추출"""
    text_no_punct = re.sub(r'[^\w\s-]', '', text)
    words = text_no_punct.split()
    proper_nouns = [word for word in words if word[0].isupper() and len(word) > 2]
    
    text_lower = text.lower()
    stop_words = [
        'a', 'an', 'the', 'what', 'how', 'who', 'when', 'where', 'why', 
        'can', 'could', 'would', 'is', 'are', 'be', 'do', 'does', 'did', 
        'in', 'of', 'for', 'to', 'and', 'or', 'it', 'its', 'their', 'by', 
        'on', 'with', 'from', 'as', 'about', 'summarize', 'outline', 
        'describe', 'propose', 'explain', 'provide', 'capture', 'distill', 
        'characterize', 'evaluate', 'summarized', 'discussed', 'based'
    ]
    
    common_words = text_lower.split()
    common_keywords = [word for word in common_words if word not in stop_words and len(word) > 3]
    
    final_keywords = list(dict.fromkeys(proper_nouns + common_keywords))
    return final_keywords[:5]

def get_verified_korean_synonyms(word: str) -> List[str]:
    """검증된 한국어 동의어 사전"""
    synonym_dict = {
        '인공지능': ['AI', 'artificial intelligence', '머신러닝', '딥러닝'],
        '머신러닝': ['machine learning', 'ML', '기계학습', '학습 알고리즘'],
        '딥러닝': ['deep learning', '신경망', 'neural network', 'CNN', 'RNN'],
        '자연어처리': ['NLP', 'natural language processing', '텍스트 분석'],
        '컴퓨터비전': ['computer vision', '이미지 처리', '영상 분석'],
        '강화학습': ['reinforcement learning', 'RL', '보상 학습'],
        '데이터마이닝': ['data mining', '데이터 분석', '패턴 발견'],
        '빅데이터': ['big data', '대용량 데이터', '데이터 처리'],
        '클라우드': ['cloud computing', '클라우드 서비스', '원격 처리'],
        '블록체인': ['blockchain', '분산원장', '암호화폐'],
        'IoT': ['internet of things', '사물인터넷', '센서 네트워크'],
        '5G': ['5th generation', '5세대', '모바일 통신'],
        '자율주행': ['autonomous driving', 'self-driving', '무인 운전'],
        '로봇': ['robot', '자동화', '메카트로닉스'],
        '드론': ['drone', 'UAV', '무인항공기'],
        '가상현실': ['VR', 'virtual reality', '증강현실', 'AR'],
        '메타버스': ['metaverse', '가상세계', '디지털 공간'],
        '암호화': ['encryption', '보안', 'cryptography'],
        '사이버보안': ['cybersecurity', '정보보안', '네트워크 보안'],
        '양자컴퓨팅': ['quantum computing', '양자 알고리즘', '양자 정보']
    }
    return synonym_dict.get(word, [word])

def extract_korean_keywords_with_synonyms(query: str, okt) -> List[str]:
    """한국어 키워드 추출 및 동의어 확장"""
    nouns = okt.nouns(query)
    important_nouns = [noun for noun in nouns if len(noun) > 1]
    
    expanded_keywords = []
    for noun in important_nouns[:3]:  # 상위 3개 명사만 확장
        synonyms = get_verified_korean_synonyms(noun)
        expanded_keywords.extend(synonyms)
    
    # 중복 제거 및 정렬
    unique_keywords = list(set(expanded_keywords))
    return unique_keywords[:8]  # 최대 8개 키워드

def extract_more_english_keywords(text: str) -> List[str]:
    """영어 텍스트에서 추가 키워드 추출 (더 넓은 범위)"""
    # 기본 키워드 추출
    basic_keywords = extract_english_keywords(text)
    
    # 추가 키워드 추출 (더 많은 불용어 포함)
    text_no_punct = re.sub(r'[^\w\s-]', '', text)
    words = text_no_punct.split()
    
    # 확장된 불용어 제거
    extended_stop_words = [
        'a', 'an', 'the', 'what', 'how', 'who', 'when', 'where', 'why', 
        'can', 'could', 'would', 'is', 'are', 'be', 'do', 'does', 'did', 
        'in', 'of', 'for', 'to', 'and', 'or', 'it', 'its', 'their', 'by', 
        'on', 'with', 'from', 'as', 'about', 'summarize', 'outline', 
        'describe', 'propose', 'explain', 'provide', 'capture', 'distill', 
        'characterize', 'evaluate', 'summarized', 'discussed', 'based',
        'also', 'into', 'only', 'then', 'more', 'most', 'even', 'must', 'may', 'might', 'shall', 'should', 'would', 'could', 'will', 'can', 'do', 'does', 'did', 'done', 'doing', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'be', 'being', 'been', 'become', 'becomes', 'becoming', 'became', 'seem', 'seems', 'seemed', 'seeming', 'appear', 'appears', 'appeared', 'appearing', 'look', 'looks', 'looked', 'looking', 'feel', 'feels', 'felt', 'feeling', 'sound', 'sounds', 'sounded', 'sounding', 'taste', 'tastes', 'tasted', 'tasting', 'smell', 'smells', 'smelled', 'smelling'
    ]
    
    # 키워드 필터링 및 정렬 (더 많은 키워드)
    keywords = [word for word in words if word.lower() not in extended_stop_words and len(word) > 1]
    
    # 빈도순 정렬 (상위 20개)
    word_freq = Counter(keywords)
    additional_keywords = [word for word, freq in word_freq.most_common(20)]
    
    # 기본 키워드와 합치고 중복 제거
    all_keywords = basic_keywords + additional_keywords
    return list(dict.fromkeys(all_keywords))  # 순서 유지하면서 중복 제거

def extract_more_korean_keywords(text: str, okt) -> List[str]:
    """한국어 텍스트에서 추가 키워드 추출 (더 넓은 범위)"""
    # 기본 키워드 추출
    basic_keywords = extract_korean_keywords_with_synonyms(text, okt)
    
    # 추가 키워드 추출 (더 많은 품사 포함)
    try:
        # 명사, 형용사, 동사 모두 추출
        pos_tags = okt.pos(text, norm=True, stem=True)
        
        # 더 많은 품사 포함
        target_pos = ['Noun', 'Adjective', 'Verb', 'Adverb']
        additional_keywords = []
        
        for word, pos in pos_tags:
            if pos in target_pos and len(word) > 1:
                additional_keywords.append(word)
        
        # 빈도순 정렬
        word_freq = Counter(additional_keywords)
        additional_keywords = [word for word, freq in word_freq.most_common(20)]
        
        # 기본 키워드와 합치고 중복 제거
        all_keywords = basic_keywords + additional_keywords
        return list(dict.fromkeys(all_keywords))  # 순서 유지하면서 중복 제거
        
    except Exception as e:
        print(f"   ⚠️  추가 한국어 키워드 추출 오류: {e}")
        return basic_keywords

def simple_semantic_filtering(documents: List[Dict], query: str) -> List[Dict]:
    """간단한 의미 기반 필터링 (키워드 매칭)"""
    if not documents:
        return []
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))
    
    # 각 문서에 점수 부여
    scored_docs = []
    for doc in documents:
        title = doc.get('title', '').lower()
        abstract = doc.get('abstract', '').lower()
        content = f"{title} {abstract}"
        
        # 키워드 매칭 점수
        content_words = set(re.findall(r'\w+', content))
        keyword_score = len(query_words.intersection(content_words)) * 2
        
        # 제목 매칭 보너스
        title_match = len(query_words.intersection(set(re.findall(r'\w+', title)))) * 3
        
        # 전문 용어 매칭 보너스
        tech_terms = ['AI', 'machine learning', 'deep learning', 'neural network', 'algorithm', 'data', 'system', 'model', 'analysis', 'research', 'study', 'method', 'approach', 'framework', 'architecture', 'technology', 'innovation', 'development', 'implementation', 'evaluation', 'performance', 'accuracy', 'efficiency', 'optimization', 'automation', 'intelligence', 'computing', 'processing', 'recognition', 'classification', 'prediction', 'forecasting', 'detection', 'monitoring', 'control', 'management', 'integration', 'deployment', 'scalability', 'robustness', 'reliability', 'security', 'privacy', 'ethics', 'sustainability', 'environmental', 'social', 'economic', 'policy', 'regulation', 'standard', 'protocol', 'interface', 'platform', 'service', 'application', 'solution', 'tool', 'software', 'hardware', 'infrastructure', 'network', 'communication', 'collaboration', 'interaction', 'user', 'experience', 'design', 'interface', 'visualization', 'representation', 'knowledge', 'information', 'data', 'database', 'storage', 'retrieval', 'search', 'query', 'indexing', 'ranking', 'filtering', 'clustering', 'segmentation', 'classification', 'regression', 'clustering', 'association', 'correlation', 'causation', 'inference', 'reasoning', 'logic', 'decision', 'planning', 'scheduling', 'optimization', 'allocation', 'distribution', 'coordination', 'synchronization', 'parallelization', 'distributed', 'centralized', 'decentralized', 'hierarchical', 'flat', 'modular', 'component', 'module', 'library', 'framework', 'api', 'sdk', 'middleware', 'backend', 'frontend', 'client', 'server', 'database', 'cache', 'queue', 'stream', 'batch', 'real-time', 'offline', 'online', 'cloud', 'edge', 'fog', 'mobile', 'web', 'desktop', 'embedded', 'iot', 'wearable', 'smart', 'intelligent', 'adaptive', 'learning', 'evolutionary', 'genetic', 'swarm', 'collective', 'emergent', 'self-organizing', 'autonomous', 'automatic', 'manual', 'semi-automatic', 'hybrid', 'multi-modal', 'cross-modal', 'inter-modal', 'trans-modal', 'meta', 'hyper', 'super', 'ultra', 'mega', 'giga', 'tera', 'peta', 'exa', 'zetta', 'yotta']
        
        tech_match = 0
        for term in tech_terms:
            if term.lower() in content:
                tech_match += 1
        
        # 문서 길이 보너스
        length_bonus = min(len(content.split()) / 100, 2)
        
        # 최종 점수 계산
        total_score = keyword_score + title_match + tech_match + length_bonus
        
        scored_docs.append((doc, total_score))
    
    # 점수순으로 정렬하고 상위 5개 선택
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:5]]

def create_kaggle_format_article(doc: Dict, index: int) -> str:
    """Kaggle 형식으로 논문 정보 생성 (title + abstract + source)"""
    title = doc.get('title', '')
    abstract = doc.get('abstract', '')
    cn = doc.get('CN', '')
    
    # Source URL 생성
    source_url = f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue=05787966&target=NART&cn={cn}"
    
    # Kaggle 형식으로 포맷팅
    formatted_article = f'Title: {title}, Abstract: {abstract}, Source: {source_url}'
    return formatted_article

def create_final_prompt_v9(query: str, context: str, language: str) -> str:
    """최종 답변 생성을 위한 프롬프트 v9 (배치 처리 최적화)"""
    language_instruction = "한국어로" if language == "ko" else "영어로 (in English)"
    
    return f"""당신은 주어진 학술 문서들을 바탕으로 질문에 대한 심층 분석 보고서를 작성하는 전문 연구원입니다.

### 참고 문서 (Context):
{context}

### 과제 (Task):
'참고 문서'의 내용을 완벽하게 숙지한 후, 아래 '출력 형식'에 맞춰 '원본 질문'에 대한 분석 보고서를 작성하세요.

### 원본 질문 (Original Question):
{query}

### 핵심 지침 (Core Directives):
1. **언어 준수**: 원본 질문이 '{language_instruction}'로 작성되었으므로, 최종 보고서 전체를 반드시 **{language_instruction}**로 작성해야 합니다.
2. **전문가의 자세**: 당신은 이 주제의 전문가입니다. "정보가 부족하다", "~일 것으로 추정된다"와 같은 불확실한 표현을 절대 사용하지 마세요.
3. **사실 기반 종합**: 여러 문서에 흩어져 있는 정보를 논리적으로 연결하고 종합하여 하나의 완성된 글로 재구성하세요.
4. **엄격한 출처 표기**: 보고서의 모든 문장은 반드시 '참고 문서'에 명시된 사실에 기반해야 합니다.

### 출력 형식 (Output Format):
1. **제목 (Title):** 질문의 핵심 내용을 포괄하는 간결하고 전문적인 제목
2. **서론 (Introduction):** 질문의 배경과 핵심 주제를 간략히 언급
3. **본론 (Body):** 참고 문서에서 찾아낸 핵심적인 사실, 데이터, 주장들을 바탕으로 구체적인 답변
4. **결론 (Conclusion):** 본론의 핵심 내용을 요약하며 보고서를 마무리

---
### 최종 보고서:
"""

def generate_fallback_answer(query: str, documents: List[Dict], language: str) -> str:
    """API 호출 실패 시 대체 답변 생성"""
    if not documents:
        return "제공된 참고 문서로는 질문에 대한 충분한 정보를 찾을 수 없습니다."
    
    # 문서에서 핵심 정보 추출
    titles = [doc.get('title', '') for doc in documents if doc.get('title')]
    
    # 간단한 템플릿 기반 답변 생성
    if language == "ko":
        answer = f"제공된 문서들을 바탕으로 '{query}'에 대한 분석을 수행했습니다.\n\n"
        answer += "주요 참고 문서:\n"
        for i, title in enumerate(titles[:3], 1):
            answer += f"{i}. {title}\n"
        answer += f"\n이 문서들은 질문과 관련된 유용한 정보를 제공합니다. 상세한 내용은 참고 문서를 확인하시기 바랍니다."
    else:
        answer = f"Based on the provided documents, I have analyzed '{query}'.\n\n"
        answer += "Key reference documents:\n"
        for i, title in enumerate(titles[:3], 1):
            answer += f"{i}. {title}\n"
        answer += f"\nThese documents provide useful information related to the question. Please refer to the documents for detailed content."
    
    return answer

def translate_text(text: str, target_language: str) -> str:
    """간단한 번역 함수 (실제로는 Gemini API 사용)"""
    # 실제 구현에서는 Gemini API를 사용하여 번역
    # 여기서는 간단한 예시로 대체
    if target_language == "en" and is_korean(text):
        # 한국어를 영어로 번역하는 간단한 예시
        return f"[Translated to English: {text}]"
    elif target_language == "ko" and not is_korean(text):
        # 영어를 한국어로 번역하는 간단한 예시
        return f"[한국어로 번역: {text}]"
    else:
        return text

def main():
    """Kaggle 제출용 RAG 파이프라인 - test.csv 형식에 맞춰 출력"""
    start_total_time = time.time()
    
    # --- 1. 초기화 ---
    print("⭐ Kaggle 제출용 RAG 파이프라인 v9.0 시작")
    
    # API 인증 정보 검증
    credentials_path = Path('./configs/scienceon_api_credentials.json')
    validate_credentials(credentials_path)
    client = ScienceONAPIClient(credentials_path=credentials_path)
    okt = Okt()
    
    # Gemini API 클라이언트 초기화
    gemini_credentials_path = Path('./configs/gemini_api_credentials.json')
    gemini_client = GeminiClient(gemini_credentials_path)
    
    print(f"   - ✅ API 클라이언트 초기화 완료")
    
    # 테스트 데이터 로드 (base DataFrame으로 사용)
    try:
        test_df = pd.read_csv("test.csv")
        print(f"\n✅ 초기화 완료. {len(test_df)}개의 질문 처리를 시작합니다.")
    except FileNotFoundError:
        print("❌ Error: test.csv file not found!")
        return
    except Exception as e:
        print(f"❌ Error reading test.csv: {e}")
        return

    # 결과 저장을 위한 메모리 리스트 초기화
    predictions = []
    predicted_articles = []

    # --- 2. [배치] 1단계: 모든 질문에 대한 문서 검색 ---
    print("\n--- [1/3] 모든 질문에 대한 문서 검색 시작 ---")
    all_questions_data = []
    
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="   - Retrieving Documents"):
        real_query = row['Question']
        question_id = index
        
        # 언어 감지 및 키워드 추출
        if is_korean(real_query):
            search_keywords = extract_korean_keywords_with_synonyms(real_query, okt)
        else:
            search_keywords = extract_english_keywords(real_query)
        
        # 모든 키워드로 검색
        all_candidate_docs = []
        for keyword in search_keywords:
            try:
                docs = client.search_articles(keyword, row_count=10, fields=['title', 'abstract', 'CN'])
                all_candidate_docs.extend(docs)
                time.sleep(0.2)  # API 호출 제한 방지
            except Exception as e:
                print(f"   ⚠️  검색 오류 (키워드: {keyword}): {e}")
                continue
        
        # 중복 제거
        unique_docs = list({doc['CN']: doc for doc in all_candidate_docs if 'CN' in doc}.values())
        
        # 50개가 안 되면 추가 키워드로 더 검색
        if len(unique_docs) < 50:
            print(f"   - 질문 {question_id+1}: {len(unique_docs)}개 문서 (50개 미만), 추가 검색 시작...")
            
            # 추가 키워드 추출 (더 넓은 범위)
            if is_korean(real_query):
                # 한국어: 더 많은 키워드 추출
                additional_keywords = extract_more_korean_keywords(real_query, okt)
            else:
                # 영어: 더 많은 키워드 추출
                additional_keywords = extract_more_english_keywords(real_query)
            
            # 기존 키워드와 중복 제거
            existing_keywords = set(search_keywords)
            new_keywords = [kw for kw in additional_keywords if kw not in existing_keywords]
            
            # 추가 검색 (최대 10개 키워드까지)
            for keyword in new_keywords[:10]:
                if len(unique_docs) >= 50:
                    break
                try:
                    docs = client.search_articles(keyword, row_count=5, fields=['title', 'abstract', 'CN'])
                    all_candidate_docs.extend(docs)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"   ⚠️  추가 검색 오류 (키워드: {keyword}): {e}")
                    continue
            
            # 다시 중복 제거
            unique_docs = list({doc['CN']: doc for doc in all_candidate_docs if 'CN' in doc}.values())
            print(f"   - 질문 {question_id+1}: 추가 검색 후 {len(unique_docs)}개 문서")
        all_questions_data.append({
            'query': real_query, 
            'id': question_id, 
            'candidates': unique_docs,
            'language': 'ko' if is_korean(real_query) else 'en'
        })
        
        print(f"   - 질문 {question_id+1}: {len(unique_docs)}개 문서 수집 완료")

    # --- 3. [배치] 2단계: 의미 기반 필터링 및 재순위화 ---
    print("\n--- [2/3] 의미 기반 필터링 및 재순위화 시작 ---")
    
    for data in tqdm(all_questions_data, desc="   - Semantic Filtering & Re-ranking"):
        if not data['candidates']:
            data['final_docs'] = []
            continue

        # 의미 기반 필터링 (상위 20개 선택)
        filtered_docs = simple_semantic_filtering(data['candidates'], data['query'])
        
        # 추가 재순위화 (상위 50개까지 확장)
        if len(filtered_docs) < 50 and len(data['candidates']) > len(filtered_docs):
            remaining_docs = [doc for doc in data['candidates'] if doc not in filtered_docs]
            # 나머지 문서들을 간단한 점수로 정렬
            remaining_scored = []
            for doc in remaining_docs:
                title = doc.get('title', '').lower()
                abstract = doc.get('abstract', '').lower()
                content = f"{title} {abstract}"
                score = len([word for word in data['query'].lower().split() if word in content])
                remaining_scored.append((doc, score))
            
            remaining_scored.sort(key=lambda x: x[1], reverse=True)
            additional_docs = [doc for doc, score in remaining_scored[:50-len(filtered_docs)]]
            filtered_docs.extend(additional_docs)
        
        # 최대 50개로 제한
        data['final_docs'] = filtered_docs[:50]
        print(f"   - 질문 {data['id']+1}: {len(data['final_docs'])}개 문서 필터링 완료")

    # --- 4. [배치] 3단계: 답변 생성 ---
    print("\n--- [3/3] 답변 생성 시작 ---")
    
    for data in tqdm(all_questions_data, desc="   - Generating Answers"):
        try:
            if not data['final_docs']:
                final_answer = "제공된 참고 문서로는 질문에 대한 충분한 정보를 찾을 수 없습니다."
            else:
                # 컨텍스트 구축 (상위 5개 문서만 사용)
                context_parts = []
                for i, doc in enumerate(data['final_docs'][:5]):
                    title = doc.get('title', '')
                    abstract = doc.get('abstract', '')
                    doc_context = f"[문서 {i+1}]\n제목: {title}\n초록: {abstract}\n"
                    context_parts.append(doc_context)
                
                context = "\n".join(context_parts)
                
                # 프롬프트 생성
                prompt_template = create_final_prompt_v9(data['query'], context, data['language'])
                
                # Gemini API로 답변 생성 (실패 시 대체 답변 사용)
                try:
                    final_answer = gemini_client.generate_answer(prompt_template)
                    
                    # 답변 품질 검증
                    if not final_answer or len(final_answer.strip()) < 20:
                        final_answer = generate_fallback_answer(data['query'], data['final_docs'], data['language'])
                        print(f"   ⚠️  Step 3: Generated answer is too short, using fallback.")
                    else:
                        print(f"   ✅ Step 3: Answer generation complete ({len(final_answer)} characters).")
                        
                except Exception as e:
                    print(f"   ⚠️  Gemini API 호출 실패: {str(e)[:100]}...")
                    final_answer = generate_fallback_answer(data['query'], data['final_docs'], data['language'])
                    print(f"   ✅ Step 3: Using fallback answer generation.")
            
            # 예측 결과를 메모리에 저장
            predictions.append(final_answer)
            
            # 상위 50개 논문 정보 추출 (test.csv 형식에 맞춰)
            article_titles = []
            for i, doc in enumerate(data['final_docs'][:50]):
                formatted_article = create_kaggle_format_article(doc, i+1)
                article_titles.append(formatted_article)
            
            # 50개가 되도록 빈 문자열로 채움
            while len(article_titles) < 50:
                article_titles.append('')
            
            predicted_articles.append(article_titles)
            
        except Exception as e:
            print(f"   ⚠️  답변 생성 오류 (질문 {data['id']+1}): {e}")
            # 오류 시 기본값으로 채움
            predictions.append(f"처리 중 오류가 발생했습니다: {str(e)}")
            predicted_articles.append([''] * 50)  # 50개 빈 문자열

    # --- 5. 최종 제출 파일 생성 (test.csv 형식에 맞춰) ---
    print("\n--- 최종 제출 파일 생성 ---")
    
    try:
        # test_df를 복사하여 base DataFrame 생성
        submission_df = test_df.copy()
        
        # Prediction 컬럼 추가
        submission_df['Prediction'] = predictions
        
        # 50개 prediction_retrieved_article_name 컬럼 추가 (모두 빈 문자열로 초기화)
        for i in range(1, 51):
            column_name = f'prediction_retrieved_article_name_{i}'
            submission_df[column_name] = [''] * len(submission_df)
        
        # 이제 각 컬럼에 실제 값 채우기
        for i in range(1, 51):
            column_name = f'prediction_retrieved_article_name_{i}'
            for row_idx, articles in enumerate(predicted_articles):
                if i-1 < len(articles) and articles[i-1]:
                    submission_df.at[row_idx, column_name] = articles[i-1]
        
        submission_path = 'submission.csv'
        # 모든 null 값을 빈 문자열로 변환
        submission_df = submission_df.fillna('')
        submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
        
        end_total_time = time.time()
        total_time = end_total_time - start_total_time
        
        print(f"\n🎉 Kaggle 제출용 파이프라인 완료!")
        print(f"   - 총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"   - 평균 처리 시간: {total_time/len(test_df):.2f}초/질문")
        print(f"   - 최종 제출 파일: {submission_path}")
        print(f"   - 파일 크기: {len(submission_df)} 행 × {len(submission_df.columns)} 열")
        
        # 성공률 계산
        successful_count = len([p for p in predictions if '오류' not in p])
        success_rate = (successful_count / len(test_df)) * 100
        print(f"   - 성공률: {successful_count}/{len(test_df)} ({success_rate:.1f}%)")
        
        # 파일 검증
        print(f"\n📊 파일 검증:")
        print(f"   - 총 질문 수: {len(submission_df)}")
        print(f"   - 답변 생성된 질문 수: {len(submission_df[submission_df['Prediction'].notna() & (submission_df['Prediction'] != '')])}")
        print(f"   - 논문 검색된 질문 수: {len(submission_df[submission_df['prediction_retrieved_article_name_1'].notna() & (submission_df['prediction_retrieved_article_name_1'] != '')])}")
        
        # null 값 확인
        null_counts = submission_df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"   ⚠️  null 값 발견: {null_counts[null_counts > 0].to_dict()}")
        else:
            print(f"   ✅ null 값 없음")
        
        # 컬럼 구조 확인
        print(f"\n📋 컬럼 구조:")
        print(f"   - 원본 컬럼 수: {len(test_df.columns)}")
        print(f"   - 최종 컬럼 수: {len(submission_df.columns)}")
        print(f"   - 추가된 컬럼: Prediction, prediction_retrieved_article_name_1 ~ prediction_retrieved_article_name_50")
        
    except Exception as e:
        print(f"❌ 제출 파일 생성 오류: {e}")

if __name__ == "__main__":
    main()
