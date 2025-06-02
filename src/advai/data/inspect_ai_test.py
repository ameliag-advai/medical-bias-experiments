import asyncio
from inspect_ai.model import get_model  
  

async def main():
    # Get a Hugging Face model  
    model = get_model("hf/mistralai/Mixtral-8x7B-Instruct-v0.1") # TinyLlama/TinyLlama-1.1B-Chat-v1.0
    
    # Send a prompt and get response  
    response = await model.generate(
        "Please fill in only the 'age' field in the following: 'Patient, {age}, {race} {sex}, presenting with: cough.'"
    )  
    
    # Print the response  
    print(f"Response: {response}")
    print(f"Response completion: {response.completion}")

asyncio.run(main())
