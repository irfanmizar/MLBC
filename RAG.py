from openai import OpenAI
import numpy as np

client = OpenAI()

with open("messi.txt", 'r') as file:
    words = file.read().split()


query_embeddings = client.embeddings.create(
    model='text-embedding-3-small', 
    input="Was Messi a great player when he was in high school?"
    ).data[0].embedding

def cosine_simularity(A, B):
    return np.dot(A,B) / ( np.linalg.norm(A) * np.linalg.norm(B) )

source_embeddings = {}
similarity_embeddings = {}
sim_keys = []
i = 0

while i < len(words):
    chunk = words[i:i + 50]
    words_chunk = ' '.join(chunk)

    response_embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=words_chunk
    ).data[0].embedding

    source_embeddings[words_chunk] = response_embeddings

    similarity = cosine_simularity(query_embeddings, response_embeddings)
    similarity_embeddings[similarity] = words_chunk
    sim_keys.append(similarity)
    print(f"Similarity: {similarity}")
    i += 50

sorted_keys = sorted(sim_keys)
top_keys = sorted_keys[:5]
chunks = []
for key in top_keys:
    temp = similarity_embeddings[key]
    print(f"Chunks: {temp}")
    chunks.append(temp)

context = " ".join(chunks)
question = "Was Messi a great player when he was in high school?"
full_prompt = f"Question: {question}\nContext: {context}"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": full_prompt
        }
      ]
    }],
    max_tokens=1000
)
print("GPT Response: \n")
print(response.choices[0].message.content)




