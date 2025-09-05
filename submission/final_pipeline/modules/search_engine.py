"""
통합 검색 엔진 (개선된 버전)
- 검색 도구 + 검색 방법 + 문서 관리자 통합
- 유연한 검색 전략 조합
- 메타데이터 기반 검색 이력 관리
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class FlexibleSearchEngine:
    """통합 검색 엔진 (개선된 버전)"""
    
    def __init__(self, document_manager):
        """
        검색 엔진 초기화
        
        Args:
            document_manager: 문서 관리자
        """
        self.document_manager = document_manager
        self.tools: Dict[str, Any] = {}
        self.methods: Dict[str, Any] = {}
        self.default_tool = None
        self.default_method = None
        
        logging.info("통합 검색 엔진 초기화 완료")
    
    def register_tool(self, tool_name: str, tool: Any, is_default: bool = False):
        """
        검색 도구 등록
        
        Args:
            tool_name: 도구 이름
            tool: 도구 인스턴스
            is_default: 기본 도구 여부
        """
        self.tools[tool_name] = tool
        if is_default:
            self.default_tool = tool_name
        
        logging.info(f"검색 도구 등록: {tool_name}")
    
    def register_method(self, method_name: str, method: Any, is_default: bool = False):
        """
        검색 방법 등록
        
        Args:
            method_name: 방법 이름
            method: 방법 인스턴스
            is_default: 기본 방법 여부
        """
        self.methods[method_name] = method
        if is_default:
            self.default_method = method_name
        
        logging.info(f"검색 방법 등록: {method_name}")
    
    def search(self, query: str, dataset_name: str = "scienceon",
               tool: Optional[str] = None, method: Optional[str] = None,
               **kwargs) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        검색 실행
        
        Args:
            query: 검색 쿼리
            dataset_name: 데이터셋 이름
            tool: 사용할 검색 도구
            method: 사용할 검색 방법
            **kwargs: 추가 옵션
            
        Returns:
            (검색 결과, 검색 메타데이터) 튜플
        """
        # 도구와 방법 결정
        tool_name = tool or self.default_tool
        method_name = method or self.default_method
        
        if not tool_name or tool_name not in self.tools:
            raise ValueError(f"등록되지 않은 검색 도구: {tool_name}")
        
        if not method_name or method_name not in self.methods:
            raise ValueError(f"등록되지 않은 검색 방법: {method_name}")
        
        # 검색 ID 생성
        search_id = self._generate_search_id(query, dataset_name, tool_name, method_name, kwargs)
        
        # 검색 메타데이터 구성
        search_metadata = {
            'search_id': search_id,
            'query': query,
            'dataset_name': dataset_name,
            'tool': tool_name,
            'method': method_name,
            'keywords': kwargs.get('keywords', []),
            'max_docs': kwargs.get('max_docs', 50),
            **kwargs
        }
        
        # 검색 실행
        start_time = datetime.now()
        
        try:
            search_method = self.methods[method_name]
            documents = search_method.search(
                query, self.tools, self.document_manager, search_metadata
            )
            
            search_time = (datetime.now() - start_time).total_seconds()
            
            # 검색 이력 저장
            self.document_manager.save_search_history(
                search_id=search_id,
                query=query,
                dataset_name=dataset_name,
                search_method=method_name,
                search_tool=tool_name,
                keywords=kwargs.get('keywords', []),
                result_count=len(documents),
                search_time=search_time,
                success=True
            )
            
            return documents, {
                'search_id': search_id,
                'tool': tool_name,
                'method': method_name,
                'search_time': search_time,
                'result_count': len(documents),
                'success': True
            }
            
        except Exception as e:
            search_time = (datetime.now() - start_time).total_seconds()
            
            # 실패한 검색 이력 저장
            self.document_manager.save_search_history(
                search_id=search_id,
                query=query,
                dataset_name=dataset_name,
                search_method=method_name,
                search_tool=tool_name,
                keywords=kwargs.get('keywords', []),
                result_count=0,
                search_time=search_time,
                success=False,
                error_message=str(e)
            )
            
            logging.error(f"검색 실행 실패: {e}")
            return [], {
                'search_id': search_id,
                'tool': tool_name,
                'method': method_name,
                'search_time': search_time,
                'result_count': 0,
                'success': False,
                'error': str(e)
            }
    
    def _generate_search_id(self, query: str, dataset_name: str, 
                           tool: str, method: str, kwargs: Dict[str, Any]) -> str:
        """검색 ID 생성"""
        content = f"{query}|{dataset_name}|{tool}|{method}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_available_tools(self) -> List[str]:
        """사용 가능한 검색 도구 목록"""
        return list(self.tools.keys())
    
    def get_available_methods(self) -> List[str]:
        """사용 가능한 검색 방법 목록"""
        return list(self.methods.keys())
    
    def get_search_statistics(self, dataset_name: str = None) -> Dict[str, Any]:
        """검색 통계 조회"""
        return self.document_manager.get_search_statistics(dataset_name)
    
    def get_document_count(self) -> int:
        """저장된 문서 수 조회"""
        return self.document_manager.get_document_count()
