"""
Agent base class - intelligent agents in the debate system
"""

from typing import List, Dict, Any, Optional
from ..utils.openai_client import OpenAIClient
from ..utils.prompt_utils import PromptUtils


class Agent:
    """Agent base class"""
    
    def __init__(self, 
                 model_name: str,
                 name: str,
                 temperature: float,
                 openai_client: OpenAIClient,
                 sleep_time: float = 0):
        """
        Initialize agent
        
        Args:
            model_name: Model name
            name: Agent name
            temperature: Temperature parameter
            openai_client: OpenAI client
            sleep_time: Request interval time
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.openai_client = openai_client
        self.sleep_time = sleep_time
        self.memory_lst: List[Dict[str, str]] = []
    
    def set_meta_prompt(self, meta_prompt: str) -> None:
        """
        Set meta prompt
        
        Args:
            meta_prompt: Meta prompt
        """
        self.memory_lst.append({
            "role": "system", 
            "content": meta_prompt
        })
    
    def add_event(self, event: str) -> None:
        """
        Add event to memory
        
        Args:
            event: Event description
        """
        self.memory_lst.append({
            "role": "user",
            "content": event
        })
    
    def add_memory(self, memory: str, verbose: bool = False) -> None:
        """
        Add memory
        
        Args:
            memory: Memory content
            verbose: Whether to print detailed information
        """
        self.memory_lst.append({
            "role": "assistant",
            "content": str(memory)
        })
    
    def ask(self, temperature: Optional[float] = None) -> Any:
        """
        Ask agent
        
        Args:
            temperature: Temperature parameter, uses default if None
            
        Returns:
            Agent response
        """
        num_context_token = sum([
            self.openai_client.num_tokens_from_string(m["content"]) 
            for m in self.memory_lst
        ])
        
        max_context = 8192
        max_token = min(max_context - num_context_token, 512)
        
        if max_token <= 0:
            raise ValueError(f"Context too long, cannot generate response. Current token count: {num_context_token}")
        
        response_text = self.openai_client.chat_completion(
            model_name=self.model_name,
            messages=self.memory_lst,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_token
        )
        
        try:
            response = PromptUtils.parse_json_response(response_text)
            if isinstance(response, dict) and len(response) > 0:
                return response
            else:
                return response_text
        except:
            return response_text
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get memory summary
        
        Returns:
            Memory summary
        """
        return {
            "name": self.name,
            "model": self.model_name,
            "memory_count": len(self.memory_lst),
            "memory": self.memory_lst
        }
    
    def clear_memory(self) -> None:
        """Clear memory"""
        self.memory_lst = []
    
    def get_last_response(self) -> Optional[str]:
        """
        Get last response
        
        Returns:
            Last response content
        """
        for message in reversed(self.memory_lst):
            if message["role"] == "assistant":
                return message["content"]
        return None


