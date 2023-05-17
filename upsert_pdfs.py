import os
from time import sleep

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

# Load PDFs and split them into chunks
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from uuid import uuid4

pdf_folder_path = './cosmos-data/'
file_names = os.listdir(pdf_folder_path)

loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, file_name)) for file_name in os.listdir(pdf_folder_path)]
datas = [loader.load() for loader in loaders]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_pdfs = [text_splitter.split_documents(data) for data in datas]

for pdf in split_pdfs:
    texts = [doc.page_content for doc in pdf]

    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done=False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass

    embeds = [record['embedding'] for record in res['data']]

    vectors = list()

    for idx, i in enumerate(texts):
        vectors.append({
            'id': str(uuid4()),
            'values': embeds[idx],
            'metadata': {
                'text': texts[idx]
            }
        })

    index.upsert(vectors=vectors)