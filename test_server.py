import openai
import asyncio
from typing import AsyncGenerator
import pytest
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 设置基础URL为本地服务器
BASE_URL = "http://localhost:8000/v1"
API_KEY = os.getenv("NOVEL_GPT_API_KEY")


@pytest.fixture
def http2_client():
    return httpx.Client(
        http2=True,
        verify=False  # 测试环境下可以禁用SSL验证
    )

def test_http2_support(http2_client):
    """测试服务器是否支持HTTP/2协议"""
    response = http2_client.get("https://localhost:8000/health")
    assert response.status_code == 200
    assert response.http_version == "HTTP/2"
    assert response.json() == {"status": "healthy"}

def test_http2_chat_completions(http2_client):
    """测试HTTP/2下的chat completions接口"""
    headers = {
        "Authorization": f"Bearer {YOUR_API_KEY}"
    }
    data = {
        "model": "nalang-xl",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False
    }
    
    response = http2_client.post(
        "https://localhost:8000/v1/chat/completions",
        headers=headers,
        json=data
    )
    assert response.status_code == 200
    assert response.http_version == "HTTP/2"

def test_health_check():
    """测试服务器健康状态"""
    print("\n测试服务器健康状态:")
    
    response = requests.get("http://localhost:8000/health")
    if response.ok:
        print("服务器状态:", response.json())
    else:
        print(f"健康检查失败: {response.status_code}")
    print()

def test_invalid_api_key():
    """测试无效的API密钥"""
    print("\n测试无效的API密钥:")
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": "Bearer invalid-key"},
        json={
            "model": "nalang-xl",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_direct_request():
    """测试直接使用requests库访问服务器"""
    print("\n测试直接HTTP请求:")
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "nalang-xl",
            "messages": [
                {"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": "你好,请介绍一下自己。"}
            ],
            "stream": True
        }
    )
    
    if response.ok:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    if line.strip() == 'data: [DONE]':
                        break
                    content = json.loads(line[6:])
                    if content.get("choices") and content["choices"][0].get("delta", {}).get("content"):
                        print(content["choices"][0]["delta"]["content"], end="", flush=True)
    else:
        print(f"请求失败: {response.status_code} - {response.text}")
    print("\n")

def test_with_openai_client():
    """测试使用OpenAI客户端访问服务器"""
    print("\n测试OpenAI客户端:")
    
    client = openai.OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    
    stream = client.chat.completions.create(
        model="nalang-xl",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "你好,请介绍一下自己。"}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    print("开始测试服务器...\n")
    print("确保服务器已经在运行 (python server.py)\n")
    
    try:
        test_health_check()
        test_invalid_api_key()
        test_direct_request()
        test_with_openai_client()
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")