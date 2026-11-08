# MOSS-TTS-Nano OpenAI Speech Service

基于 [OpenMOSS/MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) (0.1B 参数) 开发的轻量级、**纯 CPU ONNX Runtime**、**兼容 OpenAI `/v1/audio/speech` 接口**的中文优先 FastAPI 语音合成服务。

## 核心特性

- **纯 CPU 极速推理**：基于 `MOSS-TTS-Nano-100M-ONNX` 与 `MOSS-Audio-Tokenizer-Nano-ONNX`，无需安装体积庞大的 PyTorch / CUDA，单核至多核 CPU 均可流畅运行。
- **中文深度优化**：
  - 内置 6 款高品质中文音色（`Junhao`、`Zhiming`、`Weiguo`、`Xiaoyu`、`Yuewen`、`Lingyu`），并自动映射 OpenAI 标准音色名（`alloy`、`echo`、`fable`、`onyx`、`nova`、`shimmer`、`coral`）。
  - 集成 **MOSS 鲁棒文本清洗** + **WeText WFST 中文正则化 (`wetext`)**（自动转换金额、日期、小数、编号、中英混排空格与连字符）。
  - 支持长文本按标点与 Token 预算自动分句合成及句间自然停顿。
- **多级缓存加速**：
  - **L1 内存 LRU 音频缓存 + L2 磁盘持久化缓存**：相同文本/音色/语速请求 `< 1ms` 直接返回（响应头带 `X-Cache: HIT`）。
  - **句块级 (Chunk) 波形缓存**：长段落局部修改时，未修改的分句直接复用缓存波形，无需重复推理。
  - **零样本克隆参考音频编码缓存**：自定义上传或放入 `voices/` 目录的参考音频仅首次编码，特征持久化缓存。
  - **文本正则化与分词缓存**：`@lru_cache` 加速高频文本预处理。
- **OpenAI API 完整兼容**：
  - `POST /v1/audio/speech`：支持 `wav`、`mp3`、`flac`、`opus`、`aac`、`pcm` 输出格式、`speed` (0.25~4.0) 语速调节及 `stream=true` 分句流式返回。
  - `GET /v1/models`、`GET /v1/audio/voices`、`POST /v1/audio/voices`（上传自定义参考音频实现零样本声音克隆）。
- **现代 Python & 云原生部署**：基于 Python 3.13 + `uv`，内置 `Dockerfile`、`docker-compose.yml` 以及自动发布至 **GitHub Container Registry (GHCR)** 的 GitHub Actions 工作流。

---

## 音色列表与映射

| 中文/原生音色 ID | OpenAI 别名映射 | 类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `Junhao` (默认) | `alloy`, `onyx` | 中文男声 | CN 欢迎关注模思智能 |
| `Zhiming` | `echo`, `sage` | 中文男声 | CN 京味胡同闲聊 |
| `Weiguo` | `fable`, `ash` | 中文男声 | CN 说书 |
| `Xiaoyu` | `nova` | 中文女声 | CN 明星 |
| `Yuewen` | `coral` | 中文女声 | CN 机车 |
| `Lingyu` | `shimmer` | 中文女声 | CN 深夜电台 |
| `Trump`, `Ava`, `Bella`, `Adam`, `Nathan` | - | 英文音色 | 内置英文男女声 |
| `Soyo`, `Saki`, `Mortis`, `Umiri`, `Mei`, `Anon`, `Arisa` | - | 日文音色 | 内置日文女声 |

---

## 快速开始

### 1. 使用 Docker Compose 启动（推荐）

```bash
docker compose up -d
```

> **提示**：首次启动若 `./models` 目录为空，服务会自动从 HuggingFace 下载约 400MB 的 ONNX 模型文件并持久化在 `./models` 中。国内环境可在 `docker-compose.yml` 中开启 `HF_ENDPOINT=https://hf-mirror.com`。

也可以直接拉取 GHCR 镜像运行：

```bash
docker run -d \
  --name moss-tts \
  -p 8000:8000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/voices:/app/voices \
  -v $(pwd)/.cache/tts:/app/.cache/tts \
  ghcr.io/neozhu/moss-tts-speech-service:latest
```

### 2. 本地 Python 环境运行 (Python 3.12+)

```bash
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API 调用示例

### 1. 使用 OpenAI 官方 Python SDK

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-moss-tts",  # 必填，对应环境变量 MOSS_TTS_API_KEY
)

response = client.audio.speech.create(
    model="moss-tts-nano",
    voice="Xiaoyu",  # 也可用 "nova", "alloy", "Junhao", "Zhiming" 等
    input="欢迎使用模思智能语音合成服务！今天的价格是12.5元，预计2026年5月1日正式上线。",
    response_format="wav",
    speed=1.0,
)
response.write_to_file(Path("output.wav"))
```

### 2. 使用 cURL 调用

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer sk-moss-tts" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moss-tts-nano",
    "input": "你好，这是一段中文语音合成测试，支持自动缓存加速。",
    "voice": "Junhao",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### 3. 零样本声音克隆（上传自定义参考音频）

上传一段 3~10 秒清晰的参考音频即可注册新音色（自动缓存声学特征）：

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -H "Authorization: Bearer sk-moss-tts" \
  -F "voice_id=my_speaker" \
  -F "file=@./reference.wav"
```

随后在 `/v1/audio/speech` 中指定 `"voice": "my_speaker"` 即可使用该音色合成语音。

### 4. 查看与清理缓存

- 查看健康状态与缓存命中统计：`GET http://localhost:8000/health` 或 `GET http://localhost:8000/v1/cache/stats`
- 清理音频缓存：`DELETE http://localhost:8000/v1/cache`

---

## 环境变量配置 (`MOSS_TTS_*`)

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `MOSS_TTS_DEFAULT_VOICE` | `Junhao` | 默认合成音色 |
| `MOSS_TTS_CPU_THREADS` | CPU 核心数 | ONNX Runtime 线程数 |
| `MOSS_TTS_SAMPLE_MODE` | `fixed` | 采样模式：`fixed` (推荐)、`greedy` (确定性最快)、`full` |
| `MOSS_TTS_CACHE_ENABLED` | `true` | 是否开启音频与分句缓存 |
| `MOSS_TTS_CACHE_MAX_MEMORY_ITEMS` | `256` | 内存 LRU 最大缓存音频条数 |
| `MOSS_TTS_CACHE_MAX_DISK_MB` | `512` | 磁盘音频缓存上限 (MB) |
| `MOSS_TTS_API_KEY` | `sk-moss-tts` | **必填** Bearer 鉴权密钥（所有 `/v1/*` 接口强制校验） |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 下载地址（国内可设为 `https://hf-mirror.com`） |
