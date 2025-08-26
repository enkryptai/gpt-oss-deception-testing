import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
 
from vllm import LLM, SamplingParams
from datetime import datetime

class gpt_oss_20b_local:
    def __init__(self):
        self.llm = LLM(
            model="openai/gpt-oss-20b",
            trust_remote_code=True,
        )

    def convert_to_harmony_walkthrough(self, conversation, response_messages):
        """
        Convert a Harmony conversation and response messages to the walkthrough format.
        
        Args:
            conversation: Conversation object containing the input messages
            response_messages: List of Message objects from the assistant response
        
        Returns:
            str: Formatted walkthrough string
        """
        walkthrough_parts = []
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Process input conversation messages
        for message in conversation.messages:
            role = message.author.role.value
            
            if role == "system":
                # Extract system content
                system_content = message.content[0]
                model_identity = system_content.model_identity
                knowledge_cutoff = system_content.knowledge_cutoff
                reasoning_effort = system_content.reasoning_effort.lower()
                
                # Build system message
                system_msg = f"<|start|>system<|message|>{model_identity}"
                if knowledge_cutoff:
                    system_msg += f" Knowledge cutoff: {knowledge_cutoff}"
                system_msg += f" Current date: {current_date} reasoning: {reasoning_effort}"
                
                # Add channel info if available
                if hasattr(system_content, 'channel_config') and system_content.channel_config:
                    valid_channels = system_content.channel_config.valid_channels
                    system_msg += f" # Valid channels: {', '.join(valid_channels)}. Channel must be included for every message."
                    if 'functions' in str(system_content.tools) if system_content.tools else False:
                        system_msg += " Calls to these tools must go to the commentary channel: 'functions'."
                
                system_msg += "<|end|>"
                walkthrough_parts.append(system_msg)
                
            elif role == "developer":
                # Extract developer instructions
                dev_content = message.content[0]
                instructions = dev_content.instructions
                dev_msg = f"<|start|>developer<|message|># Instructions\n{instructions}<|end|>"
                walkthrough_parts.append(dev_msg)
                
            elif role == "user":
                # Extract user message
                user_content = message.content[0]
                user_text = user_content.text
                user_msg = f"<|start|>user<|message|>{user_text}<|end|>"
                walkthrough_parts.append(user_msg)
        
        # Process response messages
        for message in response_messages:
            if message.author.role.value == "assistant":
                content = message.content[0]
                text = content.text
                channel = message.channel
                
                if channel:
                    assistant_msg = f"<|start|>assistant<|channel|>{channel}<|message|>{text}<|end|>"
                else:
                    assistant_msg = f"<|start|>assistant<|message|>{text}<|end|>"
                
                walkthrough_parts.append(assistant_msg)
        
        return "".join(walkthrough_parts)


    def infer(self, system_prompt, user_prompt, reasoning_effort="Medium"):
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        new_system = SystemContent.new()
        new_system.reasoning_effort = reasoning_effort
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, new_system),
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(system_prompt),
                ),
                Message.from_role_and_content(Role.USER, user_prompt),
            ]
        )
        
        prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        
        # Harmony stop tokens (pass to sampler so they won't be included in output)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()

        sampling = SamplingParams(
            max_tokens=4096,
            temperature=1,
            stop_token_ids=stop_token_ids
        )

        outputs = self.llm.generate(
            prompt_token_ids=[prefill_ids],   # batch of size 1
            sampling_params=sampling,
        )

        gen = outputs[0].outputs[0]
        text = gen.text
        output_tokens = gen.token_ids

        entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

        harmony_walkthrough = self.convert_to_harmony_walkthrough(convo, entries)

        reasoning = entries[0].content[0].text
        response = entries[1].content[0].text

        return reasoning, response, harmony_walkthrough