"""
Debate system core module
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from .players import AffirmativePlayer, NegativePlayer, ModeratorPlayer, JudgePlayer
from ..models.claim import Claim
from ..models.evidence import EvidenceSet
from ..utils.openai_client import OpenAIClient
from ..utils.config_manager import ConfigManager
from ..utils.prompt_utils import PromptUtils
from ..utils.file_utils import FileUtils


class DebateResult:
    """Debate result class"""
    
    def __init__(self):
        self.success = False
        self.verdict = ""
        self.justification = ""
        self.debate_rounds = []
        self.participants = {}
        self.start_time = ""
        self.end_time = ""
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "verdict": self.verdict,
            "justification": self.justification,
            "debate_rounds": self.debate_rounds,
            "participants": self.participants,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata
        }


class Debate:
    """Debate system core class"""
    
    def __init__(self,
                 config: ConfigManager,
                 openai_client: OpenAIClient,
                 output_dir: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize debate system
        
        Args:
            config: Configuration manager
            openai_client: OpenAI client
            output_dir: Output directory
            verbose: Whether to show detailed information
        """
        self.config = config
        self.openai_client = openai_client
        self.output_dir = output_dir
        self.verbose = verbose
        
        self.max_rounds = config.get("debate_settings.max_rounds", 3)
        self.temperature = config.get("models.temperature", 0.9)
        self.sleep_time = config.get("api_settings.sleep_time", 0)
        
        self.debate_model = config.get("models.debate_model", "gpt-4o-mini")
        self.judge_model = config.get("models.judge_model", "gpt-4o")
        
        self.affirmative = None
        self.negative = None
        self.moderator = None
        self.judge = None
        
        self.current_claim = None
        self.current_evidence = None
        self.debate_result = DebateResult()
    
    def setup_participants(self) -> None:
        """Setup debate participants"""
        self.affirmative = AffirmativePlayer(
            model_name=self.debate_model,
            temperature=self.temperature,
            openai_client=self.openai_client,
            sleep_time=self.sleep_time
        )
        
        self.negative = NegativePlayer(
            model_name=self.debate_model,
            temperature=self.temperature,
            openai_client=self.openai_client,
            sleep_time=self.sleep_time
        )
        
        self.moderator = ModeratorPlayer(
            model_name=self.judge_model,
            temperature=self.temperature,
            openai_client=self.openai_client,
            sleep_time=self.sleep_time
        )
    
    def setup_prompts(self, claim: Claim) -> None:
        """
        Setup prompts
        
        Args:
            claim: Claim object
        """
        claim_context = claim.get_claim_context()
        evidence_text = claim.get_evidence_text()
        
        template_vars = {
            "debate_topic": claim_context,
            "evidence_set": evidence_text
        }
        
        player_meta_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.player_meta_prompt", ""), 
            template_vars
        )
        moderator_meta_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.moderator_meta_prompt", ""),
            template_vars
        )
        
        self.affirmative.set_meta_prompt(player_meta_prompt)
        self.negative.set_meta_prompt(player_meta_prompt)
        self.moderator.set_meta_prompt(moderator_meta_prompt)
    
    def run_debate(self, claim: Claim) -> DebateResult:
        """
        Run debate
        
        Args:
            claim: Claim object
            
        Returns:
            Debate result
        """
        self.debate_result.start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.current_claim = claim
        
        try:
            self.setup_participants()
            self.setup_prompts(claim)
            
            self._initialize_debate()
            
            for round_num in range(2, self.max_rounds + 1):
                if self._should_stop_debate():
                    break
                self._conduct_debate_round(round_num)
            
            if not self._should_stop_debate():
                self._final_judgment()
            
            self.debate_result.success = True
            
        except Exception as e:
            self.debate_result.success = False
            self.debate_result.metadata["error"] = str(e)
        
        self.debate_result.end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        self._save_participant_memories()
        
        return self.debate_result
    
    def _initialize_debate(self) -> None:
        """Initialize first round of debate"""
        affirmative_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.affirmative_prompt", ""),
            {
                "debate_topic": self.current_claim.get_claim_context(),
                "evidence_set": self.current_claim.get_evidence_text()
            }
        )
        
        self.affirmative.add_event(affirmative_prompt)
        aff_response = self.affirmative.ask()
        self.affirmative.add_memory(aff_response, verbose=self.verbose)
        aff_response_str = str(aff_response)
        
        negative_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.negative_prompt", ""),
            {"aff_ans": aff_response_str}
        )
        
        self.negative.add_event(negative_prompt)
        neg_response = self.negative.ask()
        self.negative.add_memory(neg_response, verbose=self.verbose)
        neg_response_str = str(neg_response)
        
        moderator_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.moderator_prompt", ""),
            {
                "aff_ans": aff_response_str,
                "neg_ans": neg_response_str,
                "round": "first"
            }
        )
        
        self.moderator.add_event(moderator_prompt)
        mod_response = self.moderator.ask()
        self.moderator.add_memory(mod_response, verbose=self.verbose)
        
        self.debate_result.debate_rounds.append({
            "round": 1,
            "affirmative_response": aff_response_str,
            "negative_response": neg_response_str,
            "moderator_response": mod_response
        })
    
    def _conduct_debate_round(self, round_num: int) -> None:
        """
        Conduct a debate round
        
        Args:
            round_num: Round number
        """
        last_round = self.debate_result.debate_rounds[-1]
        neg_ans = last_round["negative_response"]
        aff_ans = last_round["affirmative_response"]
        
        debate_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.debate_prompt", ""),
            {"oppo_ans": neg_ans}
        )
        
        self.affirmative.add_event(debate_prompt)
        aff_response = self.affirmative.ask()
        self.affirmative.add_memory(aff_response, verbose=self.verbose)
        aff_response_str = str(aff_response)
        
        debate_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.debate_prompt", ""),
            {"oppo_ans": aff_response_str}
        )
        
        self.negative.add_event(debate_prompt)
        neg_response = self.negative.ask()
        self.negative.add_memory(neg_response, verbose=self.verbose)
        neg_response_str = str(neg_response)
        
        moderator_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.moderator_prompt", ""),
            {
                "aff_ans": aff_response_str,
                "neg_ans": neg_response_str,
                "round": PromptUtils.get_round_name(round_num)
            }
        )
        
        self.moderator.add_event(moderator_prompt)
        mod_response = self.moderator.ask()
        self.moderator.add_memory(mod_response, verbose=self.verbose)
        
        self.debate_result.debate_rounds.append({
            "round": round_num,
            "affirmative_response": aff_response_str,
            "negative_response": neg_response_str,
            "moderator_response": mod_response
        })
    
    def _should_stop_debate(self) -> bool:
        """
        Determine if debate should stop
        
        Returns:
            Whether debate should stop
        """
        if not self.debate_result.debate_rounds:
            return False
        
        last_round = self.debate_result.debate_rounds[-1]
        mod_response = last_round["moderator_response"]
        
        try:
            if isinstance(mod_response, str):
                mod_response = PromptUtils.parse_json_response(mod_response)
            
            proceeding_necessity = mod_response.get("Proceeding Necessity", 
                                                  mod_response.get("Proceeding_Necessity", "Yes"))
            
            if proceeding_necessity.lower() in ["no", "n"]:
                self.debate_result.verdict = PromptUtils.validate_verdict(
                    mod_response.get("Verdict", "")
                )
                self.debate_result.justification = mod_response.get(
                    "Justification for Verdict", ""
                )
                return True
                
        except Exception:
            pass
        
        return False
    
    def _final_judgment(self) -> None:
        """Perform final judgment"""
        self.judge = JudgePlayer(
            model_name=self.judge_model,
            temperature=self.temperature,
            openai_client=self.openai_client,
            sleep_time=self.sleep_time
        )
        
        judge_meta_prompt = PromptUtils.replace_template_variables(
            self.config.get("prompts.judge_meta_prompt", ""),
            {
                "debate_topic": self.current_claim.get_claim_context(),
                "evidence_set": self.current_claim.get_evidence_text()
            }
        )
        self.judge.set_meta_prompt(judge_meta_prompt)
        
        if len(self.debate_result.debate_rounds) >= 1:
            last_round = self.debate_result.debate_rounds[-1]
            aff_final = last_round["affirmative_response"]
            neg_final = last_round["negative_response"]
        else:
            aff_final = ""
            neg_final = ""
        
        judge_prompt_1 = PromptUtils.replace_template_variables(
            self.config.get("prompts.judge_prompt_last1", ""),
            {
                "aff_ans": aff_final,
                "neg_ans": neg_final
            }
        )
        
        self.judge.add_event(judge_prompt_1)
        summary_response = self.judge.ask()
        self.judge.add_memory(summary_response, verbose=self.verbose)
        
        judge_prompt_2 = PromptUtils.replace_template_variables(
            self.config.get("prompts.judge_prompt_last2", ""),
            {"debate_topic": self.current_claim.get_claim_context()}
        )
        
        self.judge.add_event(judge_prompt_2)
        verdict_response = self.judge.ask()
        self.judge.add_memory(verdict_response, verbose=self.verbose)
        
        try:
            if isinstance(verdict_response, dict):
                verdict_data = verdict_response
            else:
                verdict_data = PromptUtils.parse_json_response(str(verdict_response))
            
            self.debate_result.verdict = PromptUtils.validate_verdict(
                verdict_data.get("Verdict", "")
            )
            self.debate_result.justification = verdict_data.get(
                "Justification for Verdict", ""
            )
            
        except Exception:
            self.debate_result.verdict = "Not Enough Evidence"
            self.debate_result.justification = "Unable to parse judgment"
    
    def _save_participant_memories(self) -> None:
        """Save participant memories"""
        participants = {}
        
        if self.affirmative:
            participants["Affirmative side"] = self.affirmative.memory_lst
        if self.negative:
            participants["Negative side"] = self.negative.memory_lst
        if self.moderator:
            participants["Moderator"] = self.moderator.memory_lst
        if self.judge:
            participants["Judge"] = self.judge.memory_lst
        
        self.debate_result.participants = participants
    
    def save_result(self, claim: Claim, exclude_keys: List[str] = None) -> None:
        """
        Save debate result
        
        Args:
            claim: Claim object
            exclude_keys: List of keys to exclude
        """
        if not self.output_dir:
            return
        
        save_data = self.debate_result.to_dict()
        save_data.update({
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "input_path": getattr(claim, 'input_path', ''),
            "model_name": self.debate_model,
            "judge_model": self.judge_model,
            "temperature": self.temperature,
            "max_rounds": self.max_rounds
        })
        
        if claim.has_evidence():
            save_data["evidence"] = claim.evidence_set.to_list()
        
        if exclude_keys:
            for key in exclude_keys:
                save_data.pop(key, None)
        
        output_file = os.path.join(self.output_dir, f"{claim.claim_id}.json")
        FileUtils.save_json(save_data, output_file)
    
    def get_result_summary(self) -> Dict[str, Any]:
        """
        Get result summary
        
        Returns:
            Result summary
        """
        return {
            "success": self.debate_result.success,
            "verdict": self.debate_result.verdict,
            "justification": self.debate_result.justification,
            "rounds_conducted": len(self.debate_result.debate_rounds),
            "start_time": self.debate_result.start_time,
            "end_time": self.debate_result.end_time
        }
