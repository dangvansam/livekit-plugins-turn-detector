# Turn detector plugin for LiveKit Agents

This plugin introduces end-of-turn detection for LiveKit Agents using custom models to determine when a user has finished speaking. It supports both built-in models and external inference servers.

Traditional voice agents use VAD (voice activity detection) for end-of-turn detection. However, VAD models lack language understanding, often causing false positives where the agent interrupts the user before they finish speaking.

By leveraging language models specifically trained for this task, this plugin offers a more accurate and robust method for detecting end-of-turns.

## Features

- **Built-in Models**: English and multilingual models that run locally
- **External Server Support**: Use custom models via OpenAI-compatible APIs or NVIDIA Triton Inference Server
- **Flexible Backends**: Choose between local inference or remote servers based on your needs
- **Async/Await**: Fully asynchronous implementation for optimal performance

See [https://docs.livekit.io/agents/build/turns/turn-detector/](https://docs.livekit.io/agents/build/turns/turn-detector/) for more information.

## Installation

### Basic Installation
```bash
pip install livekit-plugins-external-turn-detector
```

## Usage

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

For custom models or when you need to offload inference to a dedicated server, you can use external backends.

#### Using NVIDIA Triton Inference Server

For high-performance inference with custom models:

```python
from livekit.plugins.turn_detector.external import ExternalModel

turn_detector = ExternalModel(
    triton_url="localhost:7001",  # Your Triton server gRPC endpoint
    triton_model="ensemble",      # Your model name in Triton
    tokenizer="dangvansam/Qwen3-0.6B-turn-detection-en",
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

## Running your agent

This plugin requires model files. Before starting your agent for the first time, or when building Docker images for deployment, run the following command to download the model files:

```bash
python my_agent.py download-files
```

## Model system requirements

### Built-in Models

The built-in end-of-turn models are optimized to run on CPUs with modest system requirements. They are designed to run on the same server hosting your agents.

- **English model**: ~200MB RAM, ~10ms inference time
- **Multilingual model**: ~400MB RAM, ~25ms inference time
- Both models run within a shared inference server, supporting multiple concurrent sessions

### External Models

When using external backends, system requirements depend on your chosen configuration:

#### Triton Inference Server
- Server requirements depend on your model size and configuration
- Supports GPU acceleration for faster inference
- Can handle high-throughput scenarios with proper scaling
- Recommended for production deployments with custom models

## License

The plugin source code is licensed under the Apache-2.0 license.

The end-of-turn model is licensed under the [LiveKit Model License](https://huggingface.co/livekit/turn-detector/blob/main/LICENSE).
