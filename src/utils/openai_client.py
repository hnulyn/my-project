"""
OpenAI client wrapper
"""

import backoff
import time
import tiktoken
import transformers
from typing import List, Dict, Any, Optional
from openai import OpenAI, RateLimitError, APIError, APIConnectionError


# Model context length configuration
MODEL_MAX_CONTEXT = {
    "gpt-4": 7900,
    "gpt-4o-mini": 7900,
    "gpt-4.1-nano": 7900,
    "gemini-2.0-flash": 7900,
    "gpt-4o": 7900,
    "deepseek-chat": 7900,
    "o1-mini": 7900,
    "gpt-4-0314": 7900,
    "gpt-3.5-turbo-0301": 3900,
    "gpt-3.5-turbo": 3900,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
    "llama-8b-instruct": 7900,
    "llama-8b-ft": 7900,
    "llama-3.1-8B-Instruct": 7900,
    "qwen2.5-7B-Instruct": 7900,
    "llama-3.1-8b-instruct": 7900,
    "qwen2.5-7b-instruct": 7900,
}

# Supported models list
SUPPORTED_MODELS = list(MODEL_MAX_CONTEXT.keys()) + ['sft']


class OutOfQuotaException(Exception):
    """Quota exceeded exception"""
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


class AccessTerminatedException(Exception):
    """Access terminated exception"""
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


class OpenAIClient:
    """OpenAI client wrapper class"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str,
                 llama_api_key: Optional[str] = None,
                 llama_base_url: Optional[str] = None,
                 qwen_api_key: Optional[str] = None,
                 qwen_base_url: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 sleep_time: float = 0):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            llama_api_key: Llama model API key
            llama_base_url: Llama model API base URL
            qwen_api_key: Qwen model API key
            qwen_base_url: Qwen model API base URL
            cache_dir: Cache directory
            sleep_time: Request interval time
        """
        self.api_key = api_key
        self.sleep_time = sleep_time
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        if llama_api_key:
            self.llama_client = OpenAI(
                base_url=llama_base_url,
                api_key=llama_api_key
            )
        else:
            self.llama_client = None

        if qwen_api_key:
            self.qwen_client = OpenAI(
                base_url=qwen_base_url,
                api_key=qwen_api_key
            )
        else:
            self.qwen_client = None
        

    
    def num_tokens_from_string(self, text: str) -> int:
        text = self.encoding.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        num_tokens = len(self.encoding.encode(text))
        return num_tokens
    
    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=3)
    def chat_completion(self, 
                       model_name: str,
                       messages: List[Dict[str, str]],
                       temperature: float = 0.9,
                       max_tokens: int = 512,
                       top_p: float = 0.7) -> str:
        """
        Send chat completion request
        
        Args:
            model_name: Model name
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum token count
            top_p: Top-p parameter
            
        Returns:
            Generated text
        """
        time.sleep(self.sleep_time)
        
        if model_name not in SUPPORTED_MODELS and model_name.lower() not in [m.lower() for m in SUPPORTED_MODELS]:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}")
        
        if "llama" in model_name:
            if not self.llama_client:
                raise ValueError("Llama client not initialized")
            response = self.llama_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                top_p=top_p,
            )
        elif "qwen" in model_name.lower():
            if not self.qwen_client:
                raise ValueError("Qwen client not initialized")
            response = self.qwen_client.chat.completions.create(
                model="qwen2.5-7b-instruct",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                top_p=top_p,
            )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                top_p=top_p,
            )
        
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, TypeError):
            return response.choices[0].message.content


