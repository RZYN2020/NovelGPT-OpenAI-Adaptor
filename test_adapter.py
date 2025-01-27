from adapter import NovelGPTAdapter
import openai

# 测试原始 NovelGPT API
def test_novel_gpt_direct():
    print("测试原始 NovelGPT API:")
    adapter = NovelGPTAdapter()
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "你好,请介绍一下自己。"}
    ]
    
    stream = adapter.chat_completion(messages)
    for chunk in stream:
        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
    print("\n")

# 使用 OpenAI 客户端测试
def test_with_openai_client():
    print("使用 OpenAI 客户端测试:")
    
    class CustomOpenAI(openai.OpenAI):
        def __init__(self):
            super().__init__(
                base_url="https://www.gpt4novel.com/api/xiaoshuoai/ext/v1",
                api_key=NovelGPTAdapter().api_key
            )
    
    client = CustomOpenAI()
    
    response = client.chat.completions.create(
        model="nalang-xl",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "你好,请介绍一下自己。"}
        ],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    print("开始测试...\n")
    
    try:
        test_novel_gpt_direct()
        test_with_openai_client()
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")