"""
Debate participant classes
"""

from typing import Optional
from .agent import Agent
from ..utils.openai_client import OpenAIClient


class DebatePlayer(Agent):
    """Debate participant class"""
    
    def __init__(self, 
                 model_name: str,
                 name: str,
                 temperature: float,
                 openai_client: OpenAIClient,
                 sleep_time: float = 0):
        """
        Initialize debate participant
        
        Args:
            model_name: Model name
            name: Participant name
            temperature: Temperature parameter
            openai_client: OpenAI client
            sleep_time: Request interval time
        """
        super().__init__(model_name, name, temperature, openai_client, sleep_time)
        self.role = self._determine_role(name)
    
    def _determine_role(self, name: str) -> str:
        """
        Determine role based on name
        
        Args:
            name: Participant name
            
        Returns:
            Role type
        """
        name_lower = name.lower()
        if "affirmative" in name_lower or "positive" in name_lower:
            return "affirmative"
        elif "negative" in name_lower:
            return "negative"
        elif "moderator" in name_lower or "judge" in name_lower:
            return "moderator"
        else:
            return "unknown"
    
    def get_role(self) -> str:
        """
        Get participant role
        
        Returns:
            Participant role
        """
        return self.role
    
    def is_affirmative(self) -> bool:
        """
        Whether is affirmative side
        
        Returns:
            Whether is affirmative side
        """
        return self.role == "affirmative"
    
    def is_negative(self) -> bool:
        """
        Whether is negative side
        
        Returns:
            Whether is negative side
        """
        return self.role == "negative"
    
    def is_moderator(self) -> bool:
        """
        Whether is moderator
        
        Returns:
            Whether is moderator
        """
        return self.role == "moderator"


class AffirmativePlayer(DebatePlayer):
    """Affirmative debater"""
    
    def __init__(self, 
                 model_name: str,
                 temperature: float,
                 openai_client: OpenAIClient,
                 sleep_time: float = 0):
        super().__init__(model_name, "Affirmative side", temperature, openai_client, sleep_time)


class NegativePlayer(DebatePlayer):
    """Negative debater"""
    
    def __init__(self,
                 model_name: str, 
                 temperature: float,
                 openai_client: OpenAIClient,
                 sleep_time: float = 0):
        super().__init__(model_name, "Negative side", temperature, openai_client, sleep_time)


class ModeratorPlayer(DebatePlayer):
    """Moderator"""
    
    def __init__(self,
                 model_name: str,
                 temperature: float, 
                 openai_client: OpenAIClient,
                 sleep_time: float = 0):
        super().__init__(model_name, "Moderator", temperature, openai_client, sleep_time)


class JudgePlayer(DebatePlayer):
    """Judge/Judicator"""
    
    def __init__(self,
                 model_name: str,
                 temperature: float,
                 openai_client: OpenAIClient, 
                 sleep_time: float = 0):
        super().__init__(model_name, "Judge", temperature, openai_client, sleep_time)
