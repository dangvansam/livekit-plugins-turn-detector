# LiveKit Plugins – Turn Detector

**`livekit-plugins-external-turn-detector`** provides end-of-turn detection for [LiveKit Agents](https://github.com/livekit/agents) using custom models to determine when a user has finished speaking.

This plugin enables **accurate conversation flow management** by leveraging language models trained specifically for turn detection, offering superior performance compared to traditional VAD-based approaches.

## ✨ Features

- 🎯 **Built-in Models** — English and multilingual models that run locally
- 🔌 **LiveKit plugin integration** — plug-and-play support for LiveKit workflows
- 🤖 **Compatible with livekit-agents** — seamless integration with agent framework
- 🚀 **External Server Support** — use custom models via OpenAI-compatible APIs, vLLM, or NVIDIA Triton
- ⚡ **Low-latency inference** — ~10ms (English) / ~25ms (multilingual) per inference
- 🌍 **Multilingual support** — 13+ languages in the multilingual model
- 🔧 **Flexible backends** — choose between local inference or remote servers

## 🔧 Installation

```bash
# from PyPI
pip install -U livekit-plugins-external-turn-detector

# from source
pip install git+https://github.com/dangvansam/livekit-plugins-turn-detector.git
```

## 🔌 Usage

### Built-in Models

#### English model

The English model is the smaller of the two models. It requires 200MB of RAM and completes inference in ~10ms

```python
from livekit.plugins.turn_detector.english import EnglishModel

session = AgentSession(
    ...
    turn_detection=EnglishModel(),
)
```

#### Multilingual model

We've trained a separate multilingual model that supports the following languages: `English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Indonesian, Russian, Turkish`

The multilingual model requires ~400MB of RAM and completes inferences in ~25ms.

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel

session = AgentSession(
    ...
    turn_detection=MultilingualModel(),
)
```

### External Server Models

For custom models or when you need to offload inference to a dedicated server, you can use external backends. The plugin supports flexible model configuration, allowing you to use any compatible language model for turn detection.

**Supported Backends:**
- **vLLM**: High-performance inference with any HuggingFace-compatible model
- **OpenAI API**: Direct integration with OpenAI models
- **Triton**: Enterprise-grade inference server with custom model support
- **Custom APIs**: Any OpenAI-compatible API endpoint

#### Using vLLM Backend

For high-performance inference with custom models using vLLM:

```python
from livekit.plugins.turn_detector.external import ExternalModel

# Using vLLM with OpenAI-compatible API
turn_detector = ExternalModel(
    provider="openai",  # vLLM uses OpenAI-compatible API
    base_url="http://localhost:8000",  # Your vLLM server endpoint
    model_name="Qwen/Qwen3-0.6B",  # Model name in vLLM (or your custom model)
    api_key="EMPTY",  # Usually "EMPTY" for vLLM or your custom key
    temperature=0.1,
    max_tokens=20,
    system_prompt="You are a speaking turn-ending identifier. Your task is to identify whether the user's speaking turn is complete or not. Respond with 'end' if the user's turn is complete, or 'continue' if it is not."
)

session = AgentSession(
    ...
    turn_detection=turn_detector,
)
```

#### Using NVIDIA Triton Inference Server

For high-performance inference with custom models:

```python
from livekit.plugins.turn_detector.external import ExternalModel

turn_detector = ExternalModel(
    provider="triton",
    url="localhost:7001",  # Your Triton server gRPC endpoint
    model_name="ensemble",      # Your model name in Triton
    tokenizer="Qwen/Qwen3-0.6B",
    temperature=0.1,
    max_tokens=20,
)

session = AgentSession(
    ...
    turn_detection=turn_detector,
)
```

#### Using OpenAI Backend

**Environment Variables (shared across all providers):**

See [`.env.example`](./.env.example) for a complete configuration template with examples for different use cases.

**Core Configuration:**
```bash
export TURN_DETECTION_PROVIDER="openai"  # Provider: "openai" or "triton"
export TURN_DETECTION_BASE_URL="http://localhost:8000"  # Server URL
export TURN_DETECTION_MODEL="Qwen/Qwen3-0.6B"  # Any compatible model
export TURN_DETECTION_API_KEY="EMPTY"  # API key (EMPTY for vLLM, required for OpenAI)
```

**Optional Tuning Parameters:**
```bash
export TURN_DETECTION_TEMPERATURE="0.1"  # Lower = more deterministic
export TURN_DETECTION_MAX_TOKENS="20"  # Response length limit
export TURN_DETECTION_SUPPORT_LANGUAGES="en,zh"  # Target languages
export TURN_DETECTION_SYSTEM_PROMPT="Custom instructions..."  # Model behavior
export TURN_DETECTION_TOKENIZER="Qwen/Qwen3-0.6B"  # Triton only: preprocessing
```

**Flexible Model Options:**
- Use any HuggingFace model ID: `"microsoft/DialoGPT-medium"`, `"Qwen/Qwen2.5-7B-Instruct"`
- Deploy custom fine-tuned models: `"your-org/custom-turn-detector"`
- Point to local model paths with Triton or vLLM
- Configure multi-language support for your specific use case

You can then use the turn detector with just environment variables:

```python
from livekit.plugins.turn_detector.external import ExternalModel

# Using environment variables only (provider auto-detected from TURN_DETECTION_PROVIDER)
turn_detector = ExternalModel()

session = AgentSession(
    ...
    turn_detection=turn_detector,
)
```

**Easy Provider Switching:**
With unified environment variables, you can easily switch between providers:

```bash
# For vLLM/OpenAI
export TURN_DETECTION_PROVIDER="openai"
export TURN_DETECTION_BASE_URL="http://localhost:8000"
export TURN_DETECTION_MODEL="Qwen/Qwen3-0.6B"

# For Triton (same variables, different values)
export TURN_DETECTION_PROVIDER="triton"
export TURN_DETECTION_BASE_URL="localhost:7001"
export TURN_DETECTION_MODEL="ensemble"
export TURN_DETECTION_TOKENIZER="Qwen/Qwen3-0.6B"
```

#### Setting Up vLLM Server

For flexible model deployment with vLLM:

```bash
# Install vLLM
pip install vllm

# Option 1: Use Qwen models (recommended for turn detection)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000

# Option 2: Use your custom fine-tuned model
python -m vllm.entrypoints.openai.api_server \
    --model your-username/your-turn-detection-model \
    --host 0.0.0.0 \
    --port 8000

# Option 3: Local model path
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/local/model \
    --host 0.0.0.0 \
    --port 8000
```

**Model Flexibility:**
- **Any HuggingFace Model**: Use any compatible model for turn detection
- **Custom Fine-tuned Models**: Deploy your domain-specific turn detection models
- **Multi-language Support**: Configure `TURN_DETECTION_SUPPORT_LANGUAGES` for your target languages
- **Performance Tuning**: Adjust `temperature` and `max_tokens` based on your model's characteristics

#### Using NVIDIA Triton Inference Server

```python
from livekit.plugins.turn_detector.external import ExternalModel

turn_detector = ExternalModel(
    provider="triton",
    url="localhost:7001",  # Your Triton server gRPC endpoint
    model_name="ensemble",      # Your model name in Triton
    tokenizer="Qwen/Qwen3-0.6B",
    temperature=0.1,
    max_tokens=20,
)

session = AgentSession(
    ...
    turn_detection=turn_detector,
)
```

#### Triton Server Configuration

Your Triton server should have models that accept:

**Inputs:**
- `text_input` (BYTES): Input prompt
- `max_tokens` (INT32): Max tokens to generate
- `temperature` (FP32): Sampling temperature
- Additional generation parameters as needed

**Outputs:**
- `text_output` (BYTES): Generated text ("end" or "continue")

### Usage with RealtimeModel

The turn detector can be used even with speech-to-speech models such as OpenAI's Realtime API. You'll need to provide a separate STT to ensure our model has access to the text content.

```python
session = AgentSession(
    ...
    stt=deepgram.STT(model="nova-3", language="multi"),
    llm=openai.realtime.RealtimeModel(),
    turn_detection=MultilingualModel(),
)
```

## 🚀 Running your agent

This plugin requires model files. Before starting your agent for the first time, or when building Docker images for deployment, run the following command to download the model files:

```bash
python my_agent.py download-files
```

## 📊 Model system requirements

### Built-in Models

The built-in end-of-turn models are optimized to run on CPUs with modest system requirements. They are designed to run on the same server hosting your agents.

- **English model**: ~200MB RAM, ~10ms inference time
- **Multilingual model**: ~400MB RAM, ~25ms inference time
- Both models run within a shared inference server, supporting multiple concurrent sessions

### External Models

When using external backends, system requirements depend on your chosen configuration:

#### vLLM Backend
- Highly optimized for transformer models with GPU acceleration
- Supports continuous batching for improved throughput
- Memory-efficient PagedAttention for handling multiple concurrent requests
- Recommended for production deployments requiring high performance
- Compatible with most Hugging Face models

#### Triton Inference Server
- Server requirements depend on your model size and configuration
- Supports GPU acceleration for faster inference
- Can handle high-throughput scenarios with proper scaling
- Recommended for production deployments with custom models

## 📚 Documentation

For more information, see the [official documentation](https://docs.livekit.io/agents/build/turns/turn-detector/).

## 📄 License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).

## 🙏 Acknowledgments

This plugin leverages language models specifically trained for turn detection, providing more accurate conversation flow management compared to traditional VAD-based approaches.
