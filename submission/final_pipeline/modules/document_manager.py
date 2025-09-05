"""
통합 문서 관리자
- VectorDB와 MetadataManager 기능 통합
- 문서 저장, 검색, 메타데이터 관리
"""

import hashlib
import json
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

class DocumentManager:
    """통합 문서 관리자 (VectorDB + MetadataManager)"""
    
    def __init__(self, 
                 vector_db_path: str = "../vector_db",
                 metadata_db_path: str = "../data/metadata.db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "documents",
                 clear_db: bool = False):
        """
        문서 관리자 초기화
        
        Args:
            vector_db_path: 벡터 DB 경로
            metadata_db_path: 메타데이터 DB 경로
            embedding_model: 임베딩 모델명
            collection_name: 컬렉션 이름
            clear_db: DB 초기화 여부
        """
        self.vector_db_path = Path(vector_db_path)
        self.metadata_db_path = Path(metadata_db_path)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # 디렉토리 생성
        self.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DB 초기화
        if clear_db:
            self._clear_databases()
        
        # 임베딩 모델 로드
        self._load_embedding_model()
        
        # 벡터 DB 초기화
        self._init_vector_db()
        
        # 메타데이터 DB 초기화
        self._init_metadata_db()
        
        logging.info("통합 문서 관리자 초기화 완료")
    
    def _clear_databases(self):
        """데이터베이스 초기화"""
        import shutil
        
        if self.vector_db_path.exists():
            shutil.rmtree(self.vector_db_path)
            logging.info("벡터 DB 초기화 완료")
        
        if self.metadata_db_path.exists():
            self.metadata_db_path.unlink()
            logging.info("메타데이터 DB 초기화 완료")
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        logging.info(f"임베딩 모델 로딩 중... ({self.embedding_model_name})")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logging.info(f"임베딩 모델 로드 완료 (차원: {self.embedding_model.get_sentence_embedding_dimension()})")
        except Exception as e:
            logging.error(f"모델 로드 실패: {e}")
            raise
    
    def _init_vector_db(self):
        """벡터 DB 초기화"""
        try:
            self.vector_client = chromadb.PersistentClient(path=str(self.vector_db_path))
            self.vector_collection = self._get_or_create_vector_collection()
        except Exception as e:
            logging.error(f"벡터 DB 초기화 실패: {e}")
            # 인메모리 모드로 전환
            self.vector_client = chromadb.Client()
            self.vector_collection = self._get_or_create_vector_collection()
    
    def _get_or_create_vector_collection(self):
        """벡터 컬렉션 생성 또는 가져오기"""
        try:
            return self.vector_client.get_collection(self.collection_name)
        except:
            return self.vector_client.create_collection(
                name=self.collection_name,
                metadata={"description": "통합 문서 벡터 데이터베이스"}
            )
    
    def _init_metadata_db(self):
        """메타데이터 DB 초기화"""
        with sqlite3.connect(self.metadata_db_path) as conn:
            # 문서 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    content TEXT,
                    source TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 검색 이력 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    search_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    search_method TEXT NOT NULL,
                    search_tool TEXT,
                    keywords TEXT,
                    result_count INTEGER,
                    search_time REAL,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_title ON documents(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_source ON documents(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_query ON search_history(query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_dataset ON search_history(dataset_name)")
    
    def store_documents(self, documents: List[Dict], query: str = "", 
                       metadata: Dict[str, Any] = None) -> int:
        """
        문서 저장 (벡터 DB + 메타데이터 DB)
        
        Args:
            documents: 저장할 문서 리스트
            query: 검색 쿼리
            metadata: 추가 메타데이터
            
        Returns:
            저장된 문서 수
        """
        if not documents:
            return 0
        
        stored_count = 0
        
        for doc in documents:
            try:
                # 문서 ID 생성
                doc_id = self._generate_document_id(doc)
                
                # 메타데이터 DB에 저장
                if self._store_document_metadata(doc_id, doc, metadata):
                    # 벡터 DB에 저장
                    if self._store_document_vector(doc_id, doc):
                        stored_count += 1
                        
            except Exception as e:
                logging.warning(f"문서 저장 실패: {e}")
                continue
        
        logging.info(f"문서 저장 완료: {stored_count}개")
        return stored_count
    
    def _generate_document_id(self, doc: Dict) -> str:
        """문서 ID 생성"""
        # CN 필드가 있으면 사용, 없으면 제목+초록으로 해시 생성
        if 'CN' in doc and doc['CN']:
            return str(doc['CN'])
        
        content = f"{doc.get('title', '')}{doc.get('abstract', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _store_document_metadata(self, doc_id: str, doc: Dict, 
                                metadata: Dict[str, Any] = None) -> bool:
        """문서 메타데이터 저장"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doc_id, title, abstract, content, source, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    doc_id,
                    doc.get('title', ''),
                    doc.get('abstract', ''),
                    doc.get('content', ''),
                    doc.get('source', ''),
                    json.dumps(metadata or {})
                ))
                return True
        except Exception as e:
            logging.error(f"메타데이터 저장 실패: {e}")
            return False
    
    def _store_document_vector(self, doc_id: str, doc: Dict) -> bool:
        """문서 벡터 저장"""
        try:
            # 문서 텍스트 생성
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            
            # 임베딩 생성
            embedding = self.embedding_model.encode([text])[0]
            
            # 벡터 DB에 저장
            self.vector_collection.add(
                embeddings=[embedding.tolist()],
                documents=[text],
                ids=[doc_id],
                metadatas=[{
                    'title': doc.get('title', ''),
                    'abstract': doc.get('abstract', ''),
                    'source': doc.get('source', '')
                }]
            )
            return True
        except Exception as e:
            logging.error(f"벡터 저장 실패: {e}")
            return False
    
    def search_similar_documents(self, query: str, max_results: int = 50, 
                                similarity_threshold: float = 0.3) -> List[Dict]:
        """
        유사 문서 검색
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            similarity_threshold: 유사도 임계값
            
        Returns:
            유사한 문서 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 벡터 검색
            results = self.vector_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # 결과 변환
            documents = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # 거리를 유사도로 변환
                    
                    if similarity >= similarity_threshold:
                        doc = {
                            'CN': doc_id,
                            'title': results['metadatas'][0][i].get('title', ''),
                            'abstract': results['metadatas'][0][i].get('abstract', ''),
                            'source': results['metadatas'][0][i].get('source', ''),
                            'similarity': similarity
                        }
                        documents.append(doc)
            
            logging.info(f"유사 문서 검색 완료: {len(documents)}개")
            return documents
            
        except Exception as e:
            logging.error(f"유사 문서 검색 실패: {e}")
            return []
    
    def save_search_history(self, search_id: str, query: str, dataset_name: str,
                           search_method: str, search_tool: str = None,
                           keywords: List[str] = None, result_count: int = 0,
                           search_time: float = 0, success: bool = True,
                           error_message: str = None):
        """검색 이력 저장"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute("""
                    INSERT INTO search_history 
                    (search_id, query, dataset_name, search_method, search_tool, 
                     keywords, result_count, search_time, success, error_message, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    search_id, query, dataset_name, search_method, search_tool,
                    json.dumps(keywords or []), result_count, search_time, 
                    success, error_message
                ))
        except Exception as e:
            logging.error(f"검색 이력 저장 실패: {e}")
    
    def get_search_statistics(self, dataset_name: str = None, days: int = 30) -> Dict[str, Any]:
        """검색 통계 조회"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                where_clause = "created_at >= datetime('now', '-{} days')".format(days)
                params = []
                
                if dataset_name:
                    where_clause += " AND dataset_name = ?"
                    params.append(dataset_name)
                
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(search_time) as avg_search_time,
                        AVG(result_count) as avg_result_count,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_searches,
                        COUNT(CASE WHEN success = 0 THEN 1 END) as failed_searches
                    FROM search_history
                    WHERE {where_clause}
                """, params)
                
                row = cursor.fetchone()
                return {
                    'total_searches': row[0] or 0,
                    'avg_search_time': row[1] or 0,
                    'avg_result_count': row[2] or 0,
                    'successful_searches': row[3] or 0,
                    'failed_searches': row[4] or 0,
                    'success_rate': (row[3] or 0) / max(row[0] or 1, 1) * 100
                }
        except Exception as e:
            logging.error(f"검색 통계 조회 실패: {e}")
            return {}
    
    def get_document_count(self) -> int:
        """저장된 문서 수 조회"""
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"문서 수 조회 실패: {e}")
            return 0
