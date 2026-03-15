#final llm call 
#input context and user question and return answer that cannot go outside context 
#Context (facts) + Question (intent)->prompt with constraints->llm output
import os 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN=os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

client =InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

def generate_answer(context:str,question:str)->str:
    if not context.strip():
        return "I dont have enough information to answer that."
    
    SYSTEM_PROMPT = """
You are an AI assistant for an e-commerce system.

IMPORTANT RULES:
- You CANNOT see images.
- Any image information is already converted into text under 'Image description'.
- You MUST treat the image description as factual input.
- Answer strictly using the provided context.
- Do NOT say you cannot see images.
- Do NOT ask for the image again.
"""

    USER_PROMPT=f"""

Context:
{context}

Question:
{question}
"""
    
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        max_tokens=128,
        temperature=0.2
    )


    return response.choices[0].message["content"]
