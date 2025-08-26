import json
from fireworks import LLM
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from datetime import datetime

class gpt_oss_20b_fireworks:
    def __init__(self, api_key: str):
        self.client = LLM(model="gpt-oss-20b", deployment_type="auto", api_key=api_key)
        
    def convert_to_harmony_walkthrough(self, system_prompt, user_prompt, reasoning, response):
        """
        Convert system/user prompts and assistant response to harmony walkthrough format.
        
        Args:
            system_prompt: System prompt string
            user_prompt: User prompt string  
            reasoning: Reasoning content from the model
            response: Response content from the model
        
        Returns:
            str: Formatted walkthrough string
        """
        walkthrough_parts = []
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create system message
        system_msg = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2024-06 Current date: {current_date} reasoning: medium # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        walkthrough_parts.append(system_msg)
        
        # Create developer message with system prompt as instructions
        dev_msg = f"<|start|>developer<|message|># Instructions\n{system_prompt}<|end|>"
        walkthrough_parts.append(dev_msg)
        
        # Create user message
        user_msg = f"<|start|>user<|message|>{user_prompt}<|end|>"
        walkthrough_parts.append(user_msg)
        
        # Create assistant reasoning message (if reasoning exists)
        if reasoning:
            reasoning_msg = f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
            walkthrough_parts.append(reasoning_msg)
        
        # Create assistant response message
        response_msg = f"<|start|>assistant<|channel|>final<|message|>{response}<|end|>"
        walkthrough_parts.append(response_msg)
        
        return "".join(walkthrough_parts)

    def infer(self, system_prompt, user_prompt, temperature=1.0, top_p=1.0, seed=42):
        """
        Perform inference using Fireworks API.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature for sampling
            top_p: Top-p for sampling
            seed: Random seed
            
        Returns:
            tuple: (reasoning, response, harmony_walkthrough)
        """
        try:
            # Make API call
            response_obj = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )
            
            response = response_obj.choices[0].message.content
            reasoning = getattr(response_obj.choices[0].message, 'reasoning_content', None)
            
            # Generate harmony walkthrough
            harmony_walkthrough = self.convert_to_harmony_walkthrough(
                system_prompt, user_prompt, reasoning, response
            )
            
            return reasoning, response, harmony_walkthrough
            
        except Exception as e:
            raise Exception(f"Fireworks API error: {str(e)}")
