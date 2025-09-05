"""
ScienceON API 검색 도구
- ScienceON API를 통한 문서 검색
"""

import time
import logging
from typing import List, Dict, Any
from .base_tool import SearchTool

class ScienceONTool(SearchTool):
    """ScienceON API 검색 도구"""
    
    def __init__(self, api_client, config: Dict[str, Any] = None):
        """
        ScienceON 도구 초기화
        
        Args:
            api_client: ScienceON API 클라이언트
            config: 도구 설정
        """
        self.api_client = api_client
        self.config = config or {
            'min_docs': 50,
            'max_retries': 3,
            'api_delay': 0.3,
            'row_count_per_keyword': 25,
            'required_fields': ['title', 'abstract', 'CN']
        }
        self.tool_name = "scienceon"
    
    def search_documents(self, keywords: List[str], max_docs: int = 50) -> List[Dict]:
        """
        ScienceON API를 통한 문서 검색
        
        Args:
            keywords: 검색 키워드 리스트
            max_docs: 최대 문서 수
            
        Returns:
            검색된 문서 리스트
        """
        all_docs = []
        
        for keyword in keywords:
            try:
                docs = self.api_client.search_articles(
                    keyword, 
                    row_count=self.config['row_count_per_keyword'],
                    fields=self.config['required_fields']
                )
                all_docs.extend(docs)
                logging.info(f"ScienceON 키워드 '{keyword}'로 {len(docs)}개 문서 검색")
                time.sleep(self.config['api_delay'])
                
                if len(all_docs) >= max_docs:
                    break
                    
            except Exception as e:
                logging.warning(f"ScienceON 키워드 '{keyword}' 검색 실패: {e}")
                continue
        
        # 중복 제거
        unique_docs = self._remove_duplicates(all_docs)
        
        logging.info(f"ScienceON 검색 완료: {len(unique_docs)}개 문서")
        return unique_docs[:max_docs]
    
    def _remove_duplicates(self, documents: List[Dict]) -> List[Dict]:
        """중복 문서 제거"""
        seen_ids = set()
        unique_docs = []
        
        for doc in documents:
            doc_id = doc.get('CN')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs
    
    def get_tool_name(self) -> str:
        return self.tool_name
    
    def get_required_fields(self) -> List[str]:
        return self.config['required_fields']
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        self.config.update(new_config)
        logging.info(f"ScienceON 도구 설정 업데이트: {new_config}")
