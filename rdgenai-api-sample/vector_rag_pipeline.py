#!/usr/bin/env python3
"""
KURE-v1 기반 벡터 DB RAG 파이프라인
- KURE-v1 임베딩 모델 사용
- ChromaDB 벡터 데이터베이스 활용
- 한국어/영어 다국어 지원
- Kaggle 제출 형식 출력

필요한 라이브러리: requirements.txt (기존 파일 사용)
- sentence-transformers>=2.2.0
- chromadb>=0.4.0
- konlpy>=0.6.0
- 기타 기본 라이브러리들
"""

import json
import uuid
import time
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import re
from collections import Counter

# 벡터 DB 및 임베딩 라이브러리
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from konlpy.tag import Okt
    from scienceon_api_example import ScienceONAPIClient
    from gemini_client import GeminiClient
except ImportError as e:
    print(f"🚨 [오류] 필수 라이브러리가 설치되지 않았습니다: {e}")
    print("다음 명령어로 설치해주세요:")
    print("pip install -r requirements.txt")
    sys.exit(1)

class KUREVectorDB:
    """KURE-v1 기반 벡터 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "./vector_db", model_name: str = "nlpai-lab/KURE-v1"):
        """
        벡터 DB 초기화
        
        Args:
            db_path: 벡터 DB 저장 경로
            model_name: 임베딩 모델명 (기본값: KURE-v1)
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # KURE-v1 모델 로드
        print(f"🔄 KURE-v1 모델 로딩 중... ({model_name})")
        self.model = SentenceTransformer(model_name)
        print(f"✅ KURE-v1 모델 로드 완료 (차원: {self.model.get_sentence_embedding_dimension()})")
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=db_path
            )
        )
        
        # 컬렉션 생성 또는 로드
        try:
            self.collection = self.client.get_collection("papers")
            print(f"✅ 기존 벡터 DB 컬렉션 로드 완료")
        except:
            self.collection = self.client.create_collection(
                name="papers",
                metadata={"description": "학술 논문 벡터 데이터베이스"}
            )
            print(f"✅ 새로운 벡터 DB 컬렉션 생성 완료")
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        return text
    
    def create_document_text(self, doc: Dict) -> str:
        """문서를 임베딩용 텍스트로 변환"""
        title = self.preprocess_text(doc.get('title', ''))
        abstract = self.preprocess_text(doc.get('abstract', ''))
        
        # 제목과 초록을 결합 (제목에 더 높은 가중치)
        if title and abstract:
            return f"{title} [SEP] {abstract}"
        elif title:
            return title
        elif abstract:
            return abstract
        else:
            return ""
    
    def add_documents(self, documents: List[Dict]) -> int:
        """
        문서들을 벡터 DB에 추가
        
        Args:
            documents: 추가할 문서 리스트
            
        Returns:
            추가된 문서 수
        """
        if not documents:
            return 0
        
        # 기존 문서 ID 확인
        existing_ids = set()
        try:
            existing = self.collection.get()
            existing_ids = set(existing['ids'])
        except:
            pass
        
        # 새 문서만 필터링
        new_docs = []
        for doc in documents:
            doc_id = doc.get('CN', str(uuid.uuid4()))
            if doc_id not in existing_ids:
                new_docs.append(doc)
        
        if not new_docs:
            print(f"   - 모든 문서가 이미 벡터 DB에 존재합니다.")
            return 0
        
        # 문서 텍스트 생성
        texts = []
        metadatas = []
        ids = []
        
        for doc in new_docs:
            text = self.create_document_text(doc)
            if text:  # 빈 텍스트 제외
                texts.append(text)
                metadatas.append(doc)
                ids.append(doc.get('CN', str(uuid.uuid4())))
        
        if not texts:
            return 0
        
        # 임베딩 생성
        print(f"   - {len(texts)}개 문서 임베딩 생성 중...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 벡터 DB에 추가
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"   - {len(texts)}개 문서 벡터 DB 추가 완료")
        return len(texts)
    
    def search_similar(self, query: str, n_results: int = 50, threshold: float = 0.3) -> List[Dict]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            threshold: 유사도 임계값
            
        Returns:
            유사한 문서 리스트
        """
        if not query.strip():
            return []
        
        # 쿼리 전처리
        processed_query = self.preprocess_text(query)
        if not processed_query:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = self.model.encode([processed_query])
        
        # 벡터 검색
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        # 결과 필터링 및 정렬
        filtered_results = []
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            # 거리를 유사도로 변환 (ChromaDB는 거리 기반)
            similarity = 1.0 - distance
            
            if similarity >= threshold:
                metadata['similarity_score'] = similarity
                metadata['rank'] = i + 1
                filtered_results.append(metadata)
        
        return filtered_results
    
    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "model_name": self.model_name,
                "embedding_dimension": self.model.get_sentence_embedding_dimension()
            }
        except Exception as e:
            return {"error": str(e)}

class KUREVectorRAGPipeline:
    """KURE-v1 기반 벡터 RAG 파이프라인"""
    
    def __init__(self, config_path: str = "./configs"):
        """
        파이프라인 초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = Path(config_path)
        
        # API 클라이언트 초기화
        self.api_client = ScienceONAPIClient(
            credentials_path=self.config_path / "scienceon_api_credentials.json"
        )
        self.gemini_client = GeminiClient(
            credentials_path=self.config_path / "gemini_api_credentials.json"
        )
        
        # 벡터 DB 초기화
        self.vector_db = KUREVectorDB()
        
        # 한국어 처리기
        self.okt = Okt()
        
        print("✅ KURE-v1 벡터 RAG 파이프라인 초기화 완료")
    
    def extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        # 간단한 키워드 추출 (기존 로직 유지)
        if re.search('[가-힣]', query):
            # 한국어: 명사 추출
            nouns = self.okt.nouns(query)
            keywords = [noun for noun in nouns if len(noun) > 1]
            return keywords[:5]
        else:
            # 영어: 불용어 제거
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\w+', query.lower())
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            return keywords[:5]
    
    def retrieve_documents(self, query: str, max_docs: int = 100) -> List[Dict]:
        """
        문서 검색 및 벡터 DB 저장
        
        Args:
            query: 검색 쿼리
            max_docs: 최대 검색 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        print(f"🔍 문서 검색 시작: '{query[:50]}...'")
        
        # 1단계: API로 초기 검색
        keywords = self.extract_keywords(query)
        all_docs = []
        
        for keyword in keywords:
            try:
                docs = self.api_client.search_articles(
                    keyword, 
                    row_count=20, 
                    fields=['title', 'abstract', 'CN']
                )
                all_docs.extend(docs)
                time.sleep(0.2)  # API 호출 제한 방지
            except Exception as e:
                print(f"   ⚠️  키워드 '{keyword}' 검색 실패: {e}")
                continue
        
        # 중복 제거
        unique_docs = list({doc['CN']: doc for doc in all_docs if 'CN' in doc}.values())
        print(f"   - API 검색 결과: {len(unique_docs)}개 문서")
        
        # 2단계: 벡터 DB에 저장
        added_count = self.vector_db.add_documents(unique_docs)
        
        # 3단계: 벡터 유사도 검색
        similar_docs = self.vector_db.search_similar(query, n_results=max_docs)
        print(f"   - 벡터 검색 결과: {len(similar_docs)}개 문서")
        
        return similar_docs
    
    def create_context(self, documents: List[Dict], max_docs: int = 5) -> str:
        """답변 생성을 위한 컨텍스트 생성"""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents[:max_docs]):
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            similarity = doc.get('similarity_score', 0)
            
            context = f"[문서 {i+1}] (유사도: {similarity:.3f})\n"
            context += f"제목: {title}\n"
            context += f"초록: {abstract}\n"
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str, language: str) -> str:
        """답변 생성"""
        if not context:
            return "제공된 참고 문서로는 질문에 대한 충분한 정보를 찾을 수 없습니다."
        
        language_instruction = "한국어로" if language == "ko" else "영어로 (in English)"
        
        prompt = f"""당신은 주어진 학술 문서들을 바탕으로 질문에 대한 심층 분석 보고서를 작성하는 전문 연구원입니다.

### 참고 문서 (Context):
{context}

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
        
        try:
            answer = self.gemini_client.generate_answer(prompt)
            
            # 답변 품질 검증
            if not answer or len(answer.strip()) < 20:
                return self.generate_fallback_answer(query, documents, language)
            
            return answer
            
        except Exception as e:
            print(f"   ⚠️  답변 생성 실패: {e}")
            return self.generate_fallback_answer(query, documents, language)
    
    def generate_fallback_answer(self, query: str, documents: List[Dict], language: str) -> str:
        """대체 답변 생성"""
        if not documents:
            return "제공된 참고 문서로는 질문에 대한 충분한 정보를 찾을 수 없습니다."
        
        titles = [doc.get('title', '') for doc in documents[:3] if doc.get('title')]
        
        if language == "ko":
            answer = f"제공된 문서들을 바탕으로 '{query}'에 대한 분석을 수행했습니다.\n\n"
            answer += "주요 참고 문서:\n"
            for i, title in enumerate(titles, 1):
                answer += f"{i}. {title}\n"
            answer += f"\n이 문서들은 질문과 관련된 유용한 정보를 제공합니다."
        else:
            answer = f"Based on the provided documents, I have analyzed '{query}'.\n\n"
            answer += "Key reference documents:\n"
            for i, title in enumerate(titles, 1):
                answer += f"{i}. {title}\n"
            answer += f"\nThese documents provide useful information related to the question."
        
        return answer
    
    def create_kaggle_format_article(self, doc: Dict, index: int) -> str:
        """Kaggle 형식으로 논문 정보 생성"""
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        cn = doc.get('CN', '')
        
        source_url = f"http://click.ndsl.kr/servlet/OpenAPIDetailView?keyValue=05787966&target=NART&cn={cn}"
        return f'Title: {title}, Abstract: {abstract}, Source: {source_url}'
    
    def process_questions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 질문 처리
        
        Args:
            test_df: 테스트 데이터프레임
            
        Returns:
            결과 데이터프레임
        """
        print(f"🚀 {len(test_df)}개 질문 처리 시작")
        
        predictions = []
        predicted_articles = []
        
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="질문 처리"):
            query = row['Question']
            language = "ko" if re.search('[가-힣]', query) else "en"
            
            try:
                # 1단계: 문서 검색
                documents = self.retrieve_documents(query, max_docs=100)
                
                # 2단계: 컨텍스트 생성
                context = self.create_context(documents, max_docs=5)
                
                # 3단계: 답변 생성
                answer = self.generate_answer(query, context, language)
                predictions.append(answer)
                
                # 4단계: 논문 정보 추출
                article_titles = []
                for i, doc in enumerate(documents[:50]):
                    formatted_article = self.create_kaggle_format_article(doc, i+1)
                    article_titles.append(formatted_article)
                
                # 50개가 되도록 빈 문자열로 채움
                while len(article_titles) < 50:
                    article_titles.append('')
                
                predicted_articles.append(article_titles)
                
                print(f"   ✅ 질문 {index+1} 처리 완료")
                
            except Exception as e:
                print(f"   ❌ 질문 {index+1} 처리 실패: {e}")
                predictions.append(f"처리 중 오류가 발생했습니다: {str(e)}")
                predicted_articles.append([''] * 50)
        
        # 결과 데이터프레임 생성
        submission_df = test_df.copy()
        submission_df['Prediction'] = predictions
        
        # 50개 prediction_retrieved_article_name 컬럼 추가
        for i in range(1, 51):
            column_name = f'prediction_retrieved_article_name_{i}'
            submission_df[column_name] = [''] * len(submission_df)
        
        # 실제 값 채우기
        for i in range(1, 51):
            column_name = f'prediction_retrieved_article_name_{i}'
            for row_idx, articles in enumerate(predicted_articles):
                if i-1 < len(articles) and articles[i-1]:
                    submission_df.at[row_idx, column_name] = articles[i-1]
        
        return submission_df
    
    def run(self, test_file: str = "test.csv", output_file: str = "submission_kure.csv"):
        """
        전체 파이프라인 실행
        
        Args:
            test_file: 테스트 파일 경로
            output_file: 출력 파일 경로
        """
        start_time = time.time()
        
        print("⭐ KURE-v1 벡터 RAG 파이프라인 시작")
        
        # 테스트 데이터 로드
        try:
            test_df = pd.read_csv(test_file)
            print(f"✅ 테스트 데이터 로드 완료: {len(test_df)}개 질문")
        except Exception as e:
            print(f"❌ 테스트 데이터 로드 실패: {e}")
            return
        
        # 벡터 DB 통계 출력
        stats = self.vector_db.get_collection_stats()
        print(f"📊 벡터 DB 통계: {stats}")
        
        # 질문 처리
        submission_df = self.process_questions(test_df)
        
        # 결과 저장
        submission_df = submission_df.fillna('')
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 KURE-v1 벡터 RAG 파이프라인 완료!")
        print(f"   - 총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
        print(f"   - 평균 처리 시간: {total_time/len(test_df):.2f}초/질문")
        print(f"   - 최종 제출 파일: {output_file}")
        print(f"   - 파일 크기: {len(submission_df)} 행 × {len(submission_df.columns)} 열")
        
        # 성공률 계산
        successful_count = len([p for p in submission_df['Prediction'] if '오류' not in p])
        success_rate = (successful_count / len(test_df)) * 100
        print(f"   - 성공률: {successful_count}/{len(test_df)} ({success_rate:.1f}%)")

def main():
    """메인 실행 함수"""
    pipeline = KUREVectorRAGPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
