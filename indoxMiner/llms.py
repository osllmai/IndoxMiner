from abc import abstractmethod, ABC
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
    async def generate(self, prompt: str) -> str:
        pass


class OpenAi(BaseLLM):
    """OpenAI provider with enhanced error handling."""

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

    async def generate(self, prompt: str) -> str:
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


class Anthropic(BaseLLM):
    """Anthropic Claude provider with enhanced error handling."""

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

    async def generate(self, prompt: str) -> str:
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


class Ollama(BaseLLM):
    """Ollama provider with enhanced error handling and streaming support."""

    def __init__(
            self,
            model: str = "llama2",
            host: str = "http://localhost:11434"
    ):
        from ollama import AsyncClient
        self.client = AsyncClient(host=host)
        self.model = model

    async def generate(self, prompt: str) -> str:
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
