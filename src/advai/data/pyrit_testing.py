import asyncio  
from pyrit.common import initialize_pyrit, IN_MEMORY  
from pyrit.prompt_target import HuggingFaceChatTarget  
from pyrit.orchestrator import PromptSendingOrchestrator  
  
# Initialize PyRIT with in-memory storage  
initialize_pyrit(memory_db_type=IN_MEMORY)  
  
async def main():  
    # Create a Hugging Face Chat Target  
    # You'll need to set HUGGINGFACE_TOKEN environment variable  
    target = HuggingFaceChatTarget(  
        model_id="microsoft/DialoGPT-medium",  # or your preferred model  
        use_cuda=False,  # Set to True if you have CUDA available  
        max_new_tokens=50  
    )  
      
    # Create an orchestrator to send prompts  
    orchestrator = PromptSendingOrchestrator(objective_target=target)  
      
    # Send a prompt and get response  
    prompt_list = ["Hello, how are you today?"]  
    await orchestrator.send_prompts_async(prompt_list=prompt_list)  
      
    # Print the conversation  
    await orchestrator.print_conversations_async()  
  
# Run the async function  
asyncio.run(main())
