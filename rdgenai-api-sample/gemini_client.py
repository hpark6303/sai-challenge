import json
import google.generativeai as genai
from pathlib import Path
import time

class GeminiClient:
    """Gemini API 클라이언트"""
    
    def __init__(self, credentials_path: Path):
        """Gemini API 클라이언트 초기화"""
        self.credentials_path = credentials_path
        self._load_credentials()
        self._setup_client()
    
    def _load_credentials(self):
        """API 키 로드"""
        try:
            with open(self.credentials_path, "r", encoding='utf-8') as f:
                self.credentials = json.load(f)
            self.api_key = self.credentials.get("api_key")
            if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
                raise ValueError("Please set your Gemini API key in the credentials file")
        except FileNotFoundError:
            raise FileNotFoundError(f"Gemini credentials file not found: {self.credentials_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid format in Gemini credentials file: {self.credentials_path}")
    
    def _setup_client(self):
        """Gemini 클라이언트 설정"""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """답변 생성"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
                else:
                    return "답변을 생성할 수 없습니다."
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Gemini API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)  # 재시도 전 대기
                else:
                    print(f"   ❌ Gemini API 호출 최종 실패: {e}")
                    return f"API 호출 중 오류가 발생했습니다: {str(e)}"
        
        return "답변을 생성할 수 없습니다." 