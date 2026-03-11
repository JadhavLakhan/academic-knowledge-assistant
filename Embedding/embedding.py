import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )

    return response.data[0].embedding