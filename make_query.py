from dotenv import dotenv_values

config = dotenv_values('.env')

import openai
openai.api_key = config['OPENAI_API_KEY']
embed_model = 'text-embedding-ada-002'
output_dimensions = 1536

import pinecone
pinecone.init(
    api_key=config['PINECONE_API_KEY'],
    environment="us-west1-gcp-free"
)
index_name = 'simetri-knowledge-base'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=output_dimensions,
        metric='cosine'
    )
# connect to index
index = pinecone.Index(index_name)

query = 'What is the state of Cosmos ecosystem?'

limit = 3750

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=5, include_metadata=True)
    contexts = [x['metadata']['text'] for x in res['matches']]

    prompt_start = (
        'Answer the question based on the context below.\n\n' + 
        'Context:\n'
    )
    prompt_end = (
        f'\n\nQuestion: {query}\nAnswer:'
    )

    for i in range(1, len(contexts)):
        if (len('\n\n---\n\n'.join(contexts[:1])) >= limit):
            prompt = (
                prompt_start + 
                '\n\n---\n\n'.join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts) - 1:
            prompt = (
                prompt_start + 
                '\n\n---\n\n'.join(contexts) +
                prompt_end
            )

    return prompt

def complete(prompt):
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

prompt = retrieve(query)
completion = complete(prompt)
print(f'Question: {query}\nResponse: {completion}')