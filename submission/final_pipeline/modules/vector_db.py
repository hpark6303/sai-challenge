"""
벡터 데이터베이스 모듈
- ChromaDB 기반 벡터 검색
- 임베딩 모델 관리
- 문서 저장 및 검색
"""

import sys
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from .config import VECTOR_DB_CONFIG

class VectorDatabase:
    """벡터 데이터베이스 관리자"""
    
    def __init__(self, 
                 model_name: str = VECTOR_DB_CONFIG['embedding_model'],
                 db_path: str = VECTOR_DB_CONFIG['db_path'],
                 collection_name: str = VECTOR_DB_CONFIG['collection_name'],
                 clear_db: bool = False):
        """
        벡터 DB 초기화
        
        Args:
            model_name: 임베딩 모델명
            db_path: 벡터 DB 저장 경로
            collection_name: 컬렉션 이름
            clear_db: 벡터 DB 초기화 여부
        """
        self.model_name = model_name
        self.db_path = db_path
        self.collection_name = collection_name
        
        # 벡터 DB 초기화 (필요시)
        if clear_db:
            self._clear_vector_db()
        
        # 임베딩 모델 로드
        self._load_embedding_model()
        
        # ChromaDB 초기화
        self._init_chromadb()
    
    def _clear_vector_db(self):
        """벡터 DB 초기화"""
        import shutil
        import os
        
        if os.path.exists(self.db_path):
            print(f"🗑️  기존 벡터 DB 삭제 중... ({self.db_path})")
            try:
                shutil.rmtree(self.db_path)
                print(f"✅ 벡터 DB 초기화 완료")
            except Exception as e:
                print(f"⚠️  벡터 DB 삭제 실패: {e}")
        else:
            print(f"ℹ️  벡터 DB가 존재하지 않습니다.")
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        print(f"🔄 임베딩 모델 로딩 중... ({self.model_name})")
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ 임베딩 모델 로드 완료 (차원: {self.model.get_sentence_embedding_dimension()})")
        except Exception as e:
            print(f"⚠️  모델 로드 실패: {e}")
            sys.exit(1)
    
    def _init_chromadb(self):
        """ChromaDB 초기화"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self._get_or_create_collection()
        except Exception as e:
            print(f"⚠️  영구 벡터 DB 초기화 실패: {e}")
            print("🔄 인메모리 모드로 전환합니다...")
            self.client = chromadb.Client()  # 인메모리 클라이언트
            self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """컬렉션 생성 또는 가져오기"""
        try:
            return self.client.get_collection(self.collection_name)
        except:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "학술 논문 벡터 데이터베이스"}
            )
    
    def add_documents(self, documents: List[Dict]) -> int:
        """
        문서를 벡터 DB에 추가
        
        Args:
            documents: 추가할 문서 리스트
            
        Returns:
            추가된 문서 수
        """
        if not documents:
            return 0
        
        # 기존 문서 ID 확인
        existing_ids = self._get_existing_ids()
        
        # 새로운 문서만 필터링
        new_docs = self._filter_new_documents(documents, existing_ids)
        
        if not new_docs:
            print(f"   - 모든 문서가 이미 벡터 DB에 존재합니다.")
            return 0
        
        # 임베딩 생성 및 추가
        self._add_embeddings_to_db(new_docs)
        
        return len(new_docs)
    
    def _get_existing_ids(self) -> set:
        """기존 문서 ID 가져오기"""
        try:
            existing = self.collection.get()
            return set(existing['ids'])
        except:
            return set()
    
    def _filter_new_documents(self, documents: List[Dict], existing_ids: set) -> List[Dict]:
        """새로운 문서만 필터링"""
        new_docs = []
        for doc in documents:
            doc_id = str(doc.get('CN', ''))
            if doc_id and doc_id not in existing_ids:
                new_docs.append(doc)
        return new_docs
    
    def _add_embeddings_to_db(self, documents: List[Dict]):
        """임베딩을 DB에 추가"""
        try:
            # 문서 텍스트 생성
            texts = [f"{doc.get('title', '')} {doc.get('abstract', '')}" for doc in documents]
            
            # 임베딩 생성
            print(f"   - {len(texts)}개 문서 임베딩 생성 중...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # 메타데이터 및 ID 준비
            ids = [str(doc.get('CN', '')) for doc in documents]
            metadatas = [{'title': doc.get('title', ''), 'abstract': doc.get('abstract', '')} for doc in documents]
            
            # 벡터 DB에 추가
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"   - {len(texts)}개 문서 벡터 DB 추가 완료")
            
        except Exception as e:
            print(f"   ❌ 벡터 DB 추가 실패: {e}")
            # 에러가 발생해도 프로세스는 계속 진행
            if "readonly database" in str(e):
                print(f"   💡 해결 방안: vector_db 폴더 권한을 확인하고 다시 시도하세요.")
                print(f"   💡 임시 해결: 인메모리 모드로 전환합니다.")
                self._fallback_to_memory_mode()
    
    def _fallback_to_memory_mode(self):
        """읽기 전용 오류 시 인메모리 모드로 전환"""
        try:
            print("   🔄 인메모리 벡터 DB로 전환 중...")
            self.client = chromadb.Client()  # 인메모리 클라이언트
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "임시 인메모리 벡터 데이터베이스"}
            )
            print("   ✅ 인메모리 벡터 DB 초기화 완료")
        except Exception as e:
            print(f"   ❌ 인메모리 모드 전환도 실패: {e}")
            print("   ⚠️  벡터 검색 기능을 비활성화합니다.")
    
    def search_similar(self, query: str, 
                      n_results: int = VECTOR_DB_CONFIG['max_results'],
                      threshold: float = VECTOR_DB_CONFIG['similarity_threshold']) -> List[Dict]:
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
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.model.encode([query])
            
            # 벡터 검색
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['metadatas', 'distances']
            )
            
            # 결과 필터링 및 정렬
            return self._filter_search_results(results, threshold)
            
        except Exception as e:
            print(f"⚠️  벡터 검색 실패: {e}")
            return []
    
    def _filter_search_results(self, results: Dict, threshold: float) -> List[Dict]:
        """검색 결과 필터링"""
        filtered_results = []
        
        print(f"   🔍 벡터 검색 디버깅: 총 {len(results['metadatas'][0])}개 결과, 임계값: {threshold}")
        
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
            # 거리를 유사도로 변환 (개선된 방식)
            # L2 거리는 0~∞ 범위이므로, 역수 변환 후 정규화
            if distance > 0:
                similarity = 1.0 / (1.0 + distance)  # 0~1 범위로 정규화
            else:
                similarity = 1.0  # 거리가 0이면 완전 유사
            
            if similarity >= threshold:
                metadata['similarity_score'] = similarity
                metadata['rank'] = i + 1
                filtered_results.append(metadata)
            
            # 상위 5개 결과의 거리/유사도 출력
            if i < 5:
                print(f"      결과 {i+1}: 거리={distance:.4f}, 유사도={similarity:.4f}")
        
        print(f"   ✅ 필터링 후 결과: {len(filtered_results)}개")
        return filtered_results
    
    def get_stats(self) -> Dict:
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
