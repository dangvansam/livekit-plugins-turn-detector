from __future__ import annotations

import json
import os
import time
import unicodedata

from livekit.agents.inference_runner import _InferenceRunner

from .base import EOUModelBase, _EUORunnerBase
from .log import logger

_triton_config = {}


class ExternalModel(EOUModelBase):
    """
    Turn detection model using Triton Inference Server.
    """

    def __init__(
        self,
        url: str | None = None,
        model_name: str | None = None,
        tokenizer: str | None = None,
        system_prompt: str | None = None,
        unlikely_threshold: float = 0.5,
        support_languages: list[str] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 20,
        inference_executor=None,
    ):
        """
        Initialize Triton turn detection model.

        Args:
            url: Triton server gRPC URL
            model_name: Model ensemble name on Triton
            tokenizer: HuggingFace tokenizer name or instance
            system_prompt: System prompt for turn detection
            unlikely_threshold: Threshold for turn ending (0.5 = neutral)
            support_languages: List of supported language codes (default: ["en"])
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            inference_executor: Executor for running inference
        """
        super().__init__(
            model_type="external",
            inference_executor=inference_executor,
            unlikely_threshold=unlikely_threshold,
            load_languages=False,
        )

        self._support_languages = support_languages

        global _triton_config
        _triton_config = {
            "url": url,
            "model_name": model_name,
            "tokenizer": tokenizer,
            "system_prompt": system_prompt,
            "support_languages": self._support_languages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _inference_method(self) -> str:
        """Return inference method identifier."""
        return _EUORunnerExternal.INFERENCE_METHOD

    async def supports_language(self, language: str | None) -> bool:
        """Check if the model supports the given language."""
        if not language or not self._support_languages:
            return True

        if language in self._support_languages:
            return True

        return False


class _EUORunnerExternal(_EUORunnerBase):
    INFERENCE_METHOD = "external_turn_detection"

    def __init__(
        self,
        url: str | None = None,
        model_name: str | None = None,
        tokenizer: str | None = None,
        system_prompt: str | None = None,
        support_languages: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__("external")
        self._client = None
        self._np = None
        self._grpcclient = None
        self._tokenizer = None

        global _triton_config
        config = _triton_config if _triton_config else {}

        self._support_languages = (
            support_languages or
            config.get("support_languages") or
            os.getenv("TURN_DETECTION_TRITON_SUPPORT_LANGUAGES", "").split(",")
        )

        # Configuration priority: parameters > global config > environment > defaults
        self._url = url or config.get("url") or os.getenv("TURN_DETECTION_TRITON_SERVER_URL", "localhost:7001")
        self._model_name = (
            model_name or config.get("model_name") or os.getenv("TURN_DETECTION_TRITON_MODEL_NAME", "ensemble")
        )
        self._tokenizer_name = (
            tokenizer
            or config.get("tokenizer")
            or os.getenv("TURN_DETECTION_TRITON_TOKENIZER", "dangvansam/Qwen3-0.6B-turn-detection-en")
        )

        # Handle numeric parameters with proper type conversion
        if temperature is not None:
            self._temperature = temperature
        elif config.get("temperature") is not None:
            self._temperature = config["temperature"]
        else:
            self._temperature = float(os.getenv("TURN_DETECTION_TRITON_TEMPERATURE", "0.1"))

        if max_tokens is not None:
            self._max_tokens = max_tokens
        elif config.get("max_tokens") is not None:
            self._max_tokens = config["max_tokens"]
        else:
            self._max_tokens = int(os.getenv("TURN_DETECTION_TRITON_MAX_TOKENS", "20"))

        # System prompt with fallback
        default_prompt = (
            "You are a speaking turn-ending identifier. "
            "Your task is to identify whether the user's speaking turn is complete or not. "
            "Respond with `end` if the user's turn is complete, or `continue` if it is not."
        )
        self._system_prompt = (
            system_prompt
            or config.get("system_prompt")
            or os.getenv("TURN_DETECTION_TRITON_SYSTEM_PROMPT", default_prompt)
        )

    def initialize(self) -> None:
        """Initialize Triton client connection and tokenizer."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_name, truncation_side="left"
        )

        try:
            import numpy as np
            import tritonclient.grpc as grpcclient
        except ImportError as e:
            raise ImportError(
                "Triton client not installed. Install with: pip install tritonclient[grpc]"
            ) from e

        self._np = np
        self._grpcclient = grpcclient

        try:
            self._client = grpcclient.InferenceServerClient(url=self._url, verbose=False)
            if self._client.is_model_ready(self._model_name):
                logger.info(f"Model {self._model_name} is ready on {self._url}")
            else:
                logger.warning(f"Model '{self._model_name}' is not ready on {self._url}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise

    def _prepare_inputs(self, text: str, temperature: float, max_tokens: int):
        """Prepare input tensors for Triton inference."""
        inputs = []

        # Text input
        text_input = self._grpcclient.InferInput("text_input", [1, 1], "BYTES")
        text_input.set_data_from_numpy(self._np.array([[text]], dtype=object))
        inputs.append(text_input)

        # Max tokens
        max_tokens_input = self._grpcclient.InferInput("max_tokens", [1, 1], "INT32")
        max_tokens_input.set_data_from_numpy(self._np.array([[max_tokens]], dtype=self._np.int32))
        inputs.append(max_tokens_input)

        # Temperature
        temperature_input = self._grpcclient.InferInput("temperature", [1, 1], "FP32")
        temperature_input.set_data_from_numpy(
            self._np.array([[temperature]], dtype=self._np.float32)
        )
        inputs.append(temperature_input)

        # Other parameters with fixed values
        for name, value, dtype in [
            ("top_p", 0.9, "FP32"),
            ("top_k", 20, "INT32"),
            ("repetition_penalty", 1.0, "FP32"),
        ]:
            inp = self._grpcclient.InferInput(name, [1, 1], dtype)
            if dtype == "FP32":
                inp.set_data_from_numpy(self._np.array([[value]], dtype=self._np.float32))
            else:
                inp.set_data_from_numpy(self._np.array([[value]], dtype=self._np.int32))
            inputs.append(inp)

        # String parameters
        for name in ["bad_words", "stop_words"]:
            inp = self._grpcclient.InferInput(name, [1, 1], "BYTES")
            inp.set_data_from_numpy(self._np.array([[""]], dtype=object))
            inputs.append(inp)

        # Boolean parameters
        for name, value in [("exclude_input_in_output", True), ("stream", True)]:
            inp = self._grpcclient.InferInput(name, [1, 1], "BOOL")
            inp.set_data_from_numpy(self._np.array([[value]], dtype=bool))
            inputs.append(inp)

        return inputs

    def _run_inference_sync(self, prompt: str) -> str:
        """Run synchronous inference on Triton server."""
        accumulated_text = []

        def stream_callback(result, error):
            if error:
                logger.error(f"Stream error: {error}")
            else:
                chunk = result.as_numpy("text_output")[0].decode()
                accumulated_text.append(chunk)

        try:
            inputs = self._prepare_inputs(prompt, self._temperature, self._max_tokens)
            outputs = [self._grpcclient.InferRequestedOutput("text_output")]

            self._client.start_stream(callback=stream_callback)
            self._client.async_stream_infer(
                model_name=self._model_name,
                inputs=inputs,
                outputs=outputs,
            )
            self._client.stop_stream()

            return "".join(accumulated_text).strip().lower()

        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            if self._client:
                self._client.stop_stream()
            raise

    def run(self, data: bytes) -> bytes | None:
        """Run inference using Triton server."""
        data_json = json.loads(data)
        chat_ctx = data_json.get("chat_ctx", None)

        if not chat_ctx:
            raise ValueError("chat_ctx is required on the inference input data")

        start_time = time.perf_counter()

        # Process chat context to extract last user message
        last_user_content = []
        for item in reversed(chat_ctx):
            if item["role"] == "user":
                text_content = item["content"]
                text_content = unicodedata.normalize("NFKC", text_content).strip()
                # text_content = text_content.lower()
                # for punct in ".,?!":
                #     text_content = text_content.replace(punct, "")
                if text_content:
                    last_user_content.insert(0, text_content)
            elif item["role"] == "assistant" and last_user_content:
                break

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        if last_user_content:
            last_user_content = last_user_content[:3]
            messages.append({"role": "user", "content": " ".join(last_user_content)})
        else:
            # No user content, return 0 probability
            result = {
                "eou_probability": 0.0,
                "input": "",
                "duration": 0.0,
            }
            return json.dumps(result).encode()

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=True,
            enable_thinking=False,
        )
        result_text = self._run_inference_sync(prompt)

        eou_probability = 1.0 if "end" in result_text else 0.0

        end_time = time.perf_counter()

        result = {
            "eou_probability": float(eou_probability),
            "status": result_text,
            "duration": round(end_time - start_time, 3),
            "input": " ".join(last_user_content),
            # "messages": messages,
        }

        return json.dumps(result).encode()


_InferenceRunner.register_runner(_EUORunnerExternal)
