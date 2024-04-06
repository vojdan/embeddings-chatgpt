import config
from openai import OpenAI

client = OpenAI()

# SOURCE
# https://platform.openai.com/docs/guides/embeddings

# MODELS
# text-embedding-3-small
# text-embedding-3-large

# SAMPLE EMBEDDING RESPONSE
# {
#   "object": "list",
#   "data": [
#     {
#       "object": "embedding",
#       "index": 0,
#       "embedding": [
#         -0.006929283495992422,
#         -0.005336422007530928,
#         ... (omitted for spacing)
#         -4.547132266452536e-05,
#         -0.024047505110502243
#       ],
#     }
#   ],
#   "model": "text-embedding-3-small",
#   "usage": {
#     "prompt_tokens": 5,
#     "total_tokens": 5
#   }
# }


# a function to get the embedding of a text string
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


# a function to get the embedding for the amazon fine food reviews
def get_embedding_batch(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding
