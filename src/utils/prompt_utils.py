"""
Prompt processing utilities
"""

import json
import json_repair
from typing import Dict, Any, Union


class PromptUtils:
    """Prompt processing utility class"""
    
    @staticmethod
    def replace_template_variables(template: Union[str, Dict[str, Any]], 
                                 variables: Dict[str, str]) -> Union[str, Dict[str, Any]]:
        """
        Replace variables in template
        
        Args:
            template: Template string or dictionary
            variables: Variable dictionary
            
        Returns:
            Replaced template
        """
        if isinstance(template, str):
            result = template
            for key, value in variables.items():
                result = result.replace(f"##{key}##", str(value))
            return result
        
        elif isinstance(template, dict):
            json_str = json.dumps(template)
            for key, value in variables.items():
                json_str = json_str.replace(f"##{key}##", str(value))
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return json_repair.loads(json_str)
        
        else:
            return template
    
    @staticmethod
    def format_evidence_for_prompt(evidence_list: list) -> str:
        """
        Format evidence list for prompt text
        
        Args:
            evidence_list: Evidence list
            
        Returns:
            Formatted evidence text
        """
        if not evidence_list:
            return "No evidence available."
        
        formatted_evidence = []
        for i, evidence in enumerate(evidence_list):
            if isinstance(evidence, dict):
                question = evidence.get('question', '')
                answer = evidence.get('answer', '')
                url = evidence.get('url', '')
                
                formatted_evidence.append(
                    f"Evidence_ID: {i}\n"
                    f"Evidence_Content:{{Question: {question}\n"
                    f"Answer: {answer}}}\n"
                    f"Evidence_URL: {url}"
                )
            else:
                formatted_evidence.append(f"Evidence_ID: {i}\nEvidence_Content: {evidence}")
        
        return "\n".join(formatted_evidence)
    
    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """
        Parse JSON format response
        
        Args:
            response: Response string
            
        Returns:
            Parsed dictionary
        """
        if isinstance(response, dict):
            return response
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                return json_repair.loads(response)
            except:
                try:
                    result = eval(response)
                    if isinstance(result, dict):
                        return result
                except:
                    pass
                
                return PromptUtils._extract_json_from_text(response)
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, Any]:
        """
        Extract JSON part from text
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON dictionary
        """
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = text[start_idx:end_idx+1]
            try:
                return json.loads(json_str)
            except:
                try:
                    return json_repair.loads(json_str)
                except:
                    pass
        
        return {"error": "Unable to parse JSON response", "raw_response": text}
    
    @staticmethod
    def validate_verdict(verdict: str) -> str:
        """
        Validate and normalize verdict result
        
        Args:
            verdict: Original verdict
            
        Returns:
            Normalized verdict
        """
        valid_verdicts = [
            "Supported",
            "Refuted",
            "Not Enough Evidence", 
            "Conflicting Evidence/Cherrypicking"
        ]
        
        verdict = verdict.strip()
        
        if verdict in valid_verdicts:
            return verdict
        
        verdict_lower = verdict.lower()
        if "support" in verdict_lower:
            return "Supported"
        elif "refut" in verdict_lower:
            return "Refuted"
        elif "not enough" in verdict_lower or "insufficient" in verdict_lower:
            return "Not Enough Evidence"
        elif "conflict" in verdict_lower or "cherry" in verdict_lower:
            return "Conflicting Evidence/Cherrypicking"
        
        return "Not Enough Evidence"
    
    @staticmethod
    def get_round_name(round_num: int) -> str:
        """
        Get round name
        
        Args:
            round_num: Round number
            
        Returns:
            Round name
        """
        round_names = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }
        return round_names.get(round_num, f"{round_num}th")
    
    @staticmethod
    def clean_response(response: str) -> str:
        """
        Clean response text
        
        Args:
            response: Original response
            
        Returns:
            Cleaned response
        """
        if isinstance(response, dict):
            return str(response)
        
        response = response.strip()
        
        if response.startswith('```') and response.endswith('```'):
            lines = response.split('\n')
            if len(lines) > 2:
                response = '\n'.join(lines[1:-1])
        
        return response
