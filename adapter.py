import os
import json
from typing import Iterator, Optional, Union, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

class NovelGPTAdapter:
    def __init__(self):
        self.api_key = os.getenv("NOVEL_GPT_API_KEY")
        self.api_base = "https://www.gpt4novel.com/api/xiaoshuoai/ext/v1"
        
    def chat_completion(
        self,
        messages: list,
        model: str = "nalang-xl",
        temperature: float = 0.7,
        max_tokens: int = 800,
        top_p: float = 0.35,
        stream: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """OpenAI 兼容的聊天完成接口"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            "repetition_penalty": kwargs.get("repetition_penalty", 1.05)
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=data,
            stream=stream
        )
        
        if not response.ok:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
            
        if not stream:
            return response.json()
            
        def generate_stream() -> Iterator[Dict[str, Any]]:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_data = json.loads(line[6:])
                        yield json_data
                        
        return generate_stream()