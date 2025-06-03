import pandas as pd

from langfair.utils.dataloader import load_realtoxicity
from langfair.generator import CounterfactualGenerator

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_vertexai import ChatVertexAI


rl = InMemoryRateLimiter(
  requests_per_second=4,
  check_every_n_seconds=0.5,
  max_bucket_size=240,
)
llm = ChatVertexAI(model_name='gemini-pro', temperature=0.3, rate_limiter=rl)


# INSTRUCTION = "Summarize the following conversation in no more than 3 sentences: \n"
# with open('src/advai/data/neil_code_dialogsum_train.txt', 'r') as file:
#     prompts = [INSTRUCTION + str(line) for line in file]


# rg = ResponseGenerator(langchain_llm=llm)
# generations = await rg.generate_responses(prompts=prompts, count=25)

n=50000 # number of prompts we want to test
prompts = load_realtoxicity(n=n)
print(f"\nExample prompt\n{'-'*14}\n{prompts[0]}")

suppressed_exceptions = (IndexError, )
cdg = CounterfactualGenerator(
    langchain_llm=llm,
    suppressed_exceptions=suppressed_exceptions
)

# Check for fairness through unawareness
ftu_result = cdg.check_ftu(
    prompts=prompts,
    attribute='race',
    subset_prompts=True
)
race_prompts = pd.DataFrame(ftu_result["data"]).rename(columns={'attribute_words': 'race_words'})

prompts_dict = cdg.create_prompts(prompts=race_prompts['prompt'], attribute='race')

