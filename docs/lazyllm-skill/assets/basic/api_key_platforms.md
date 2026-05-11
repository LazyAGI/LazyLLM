# 在线模型平台与 API Key 环境变量

使用 OnlineModule 或 AutoModel 调用线上模型前，需在对应平台申请 API Key，并设置下列环境变量（断网环境下仅作配置参考；实际申请需联网访问各平台）。

## 通用格式

```bash
export LAZYLLM_<平台环境变量名>_API_KEY=<你的 key>
```

## 平台与环境变量

| 平台       | 环境变量 | 申请/文档链接 |
|------------|----------|---------------|
| 日日新 SenseNova | `LAZYLLM_SENSENOVA_API_KEY`，可选 `LAZYLLM_SENSENOVA_SECRET_KEY` | [平台与 API 申请](https://platform.sensenova.cn/) |
| OpenAI     | `LAZYLLM_OPENAI_API_KEY` | [API Keys](https://platform.openai.com/api-keys) |
| 智谱 GLM   | `LAZYLLM_GLM_API_KEY` | [开放平台](https://open.bigmodel.cn/) |
| Kimi       | `LAZYLLM_KIMI_API_KEY` | [开放平台](https://platform.moonshot.cn/) |
| 通义千问 Qwen | `LAZYLLM_QWEN_API_KEY` | [阿里云百炼](https://bailian.console.aliyun.com/) |
| 豆包 Doubao | `LAZYLLM_DOUBAO_API_KEY` | [火山引擎](https://console.volcengine.com/ark) |
| 硅基流动 SiliconFlow | `LAZYLLM_SILICONFLOW_API_KEY` | [SiliconFlow](https://cloud.siliconflow.cn/) |
| MiniMax    | `LAZYLLM_MINIMAX_API_KEY` | [MiniMax 开放平台](https://api.minimax.chat/) |
| AI Ping    | `LAZYLLM_AIPING_API_KEY` | - |
| DeepSeek   | `LAZYLLM_DEEPSEEK_API_KEY` | [DeepSeek 开放平台](https://platform.deepseek.com/) |

说明：日日新可使用“仅 API Key”或“AK + SK”两种方式；其他平台一般只需配置对应 `_API_KEY`。完整列表与申请方式见项目主文档（需联网时查阅）。
