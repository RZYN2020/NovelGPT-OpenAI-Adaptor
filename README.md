# NovelGPT OpenAI 兼容适配器

这个项目提供了一个将 NovelGPT API 转换为 OpenAI 兼容接口的适配器服务。它允许你使用标准的 OpenAI 客户端库来访问 NovelGPT 的服务。

## 功能特点

- ✨ 完全兼容 OpenAI API 格式
- 🚀 支持流式输出
- 🔑 API 密钥认证
- 🔄 保留原始 API 的所有参数
- 🛡️ 内置错误处理
- 🔍 健康检查端点

## 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/novel-gpt-adapter.git
cd novel-gpt-adapter
```

2. 创建并激活虚拟环境：

```bash
python -m venv venv
source venv/bin/activate # Windows使用: venv\Scripts\activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 创建 `.env` 文件并设置你的 API 密钥：

```env
NOVEL_GPT_API_KEY=你的NovelGPT API密钥
```

5. 启动服务器

```bash
python server.py
```

