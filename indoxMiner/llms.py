from abc import ABC, abstractmethod
import requests
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


class BaseLLM(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str) -> str:
        pass


class AsyncOpenAi(BaseLLM):
    """Asynchronous OpenAI provider with enhanced error handling."""

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            temperature: float = 0.0,
            max_tokens: int = 2000,
            base_url: str = None
    ):
        from openai import AsyncOpenAI
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_async(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OpenAi(BaseLLM):
    """Synchronous OpenAI provider with enhanced error handling."""

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            temperature: float = 0.0,
            max_tokens: int = 2000,
            base_url: str = None
    ):
        from openai import OpenAI
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class AsyncAnthropic(BaseLLM):
    """Asynchronous Anthropic Claude provider with enhanced error handling."""

    def __init__(
            self,
            api_key: str,
            model: str = "claude-3-opus-20240229",
            temperature: float = 0.0,
            max_tokens: int = 2000
    ):
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_async(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class Anthropic(BaseLLM):
    """Synchronous Anthropic Claude provider with enhanced error handling."""

    def __init__(
            self,
            api_key: str,
            model: str = "claude-3-opus-20240229",
            temperature: float = 0.0,
            max_tokens: int = 2000
    ):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise


class AsyncOllama(BaseLLM):
    """Asynchronous Ollama provider with enhanced error handling and streaming support."""

    def __init__(
            self,
            model: str = "llama2",
            host: str = "http://localhost:11434"
    ):
        from ollama import AsyncClient
        self.client = AsyncClient(host=host)
        self.model = model

    async def generate_async(self, prompt: str) -> str:
        """
        Generates a response from the Ollama model asynchronously.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            Exception: If the generation fails.
        """
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
            )
            result = response['response'].strip()
            return result
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise


class Ollama(BaseLLM):
    """Synchronous Ollama provider with enhanced error handling and streaming support."""

    def __init__(
            self,
            model: str = "llama2",
            host: str = "http://localhost:11434"
    ):
        from ollama import Client
        self.client = Client(host=host)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the Ollama model synchronously.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.

        Raises:
            Exception: If the generation fails.
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
            )
            result = response['response'].strip()
            return result
        except Exception as e:
            logger.error(f"Ollama completion failed: {e}")
            raise


class AsyncIndoxApi(BaseLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate_async(self,
                             prompt: str,
                             system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
                             max_tokens: int = 4000,
                             temperature: float = 0.3,
                             stream: bool = True,
                             model: str = "gpt-4o-mini",
                             presence_penalty: float = 0,
                             frequency_penalty: float = 0,
                             top_p: float = 1):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": prompt, "role": "user"}
            ],
            "model": model,
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p
        }

        response = await requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = await response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")


class IndoxApi(BaseLLM):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self,
                 prompt: str,
                 system_prompt: str = "You are a precise data extraction assistant. Extract exactly what is asked for, nothing more.",
                 max_tokens: int = 4000,
                 temperature: float = 0.3,
                 stream: bool = True,
                 model: str = "gpt-4o-mini",
                 presence_penalty: float = 0,
                 frequency_penalty: float = 0,
                 top_p: float = 1):
        url = 'http://5.78.55.161/api/chat_completion/generate/'
        headers = {
            'accept': '*/*',
            "Authorization": f"Bearer {self.api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "messages": [
                {"content": system_prompt, "role": "system"},
                {"content": prompt, "role": "user"}
            ],
            "model": model,
            "presence_penalty": presence_penalty,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer_data = response.json()
            generated_text = answer_data.get("text_message", "")
            return generated_text
        else:
            raise Exception(f"Error From Indox API: {response.status_code}, {response.text}")
