import json
import numpy as np
from FlagEmbedding import FlagAutoModel
import time 
from sklearn.metrics.pairwise import cosine_similarity
import os

def main():
    start_time = time.time()

    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-base-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        #  devices='cpu', # Uncomment this line if you want to use GPU.
         use_fp16=True)

    model_end = time.time()
    model_loading_time = model_end - start_time
    print('model_loading_time',model_loading_time)
    with open('PATH_TO_YOUR_JSON.json', 'r', encoding='utf-8') as file:
       new_data = json.load(file)
    embed_start = time.time()
    to_embed = []
    for document in new_data:
       content = document.get('content', '')  
       to_embed.append(content.strip()) 
    
    embeddings_np = np.array(model.encode(to_embed))
    
    
    # embeddings_np=np.load('PATH_TO_YOUR_NPY_FILE') #Use this line if you want to load precomputed embeddings.
    
    
    np.save('PATH_TO_YOUR_EMBEDDING.npy', embeddings_np)
    embed_end = time.time()
    embedding_time = embed_end - embed_start
    print('embedding_time:',embedding_time)



#Test demo with simple KNN cosine_similarity
    # query='This is a test query to find relevant documents.'
    # query_embedding=np.array(model.encode(query))
    # query_embedding_end=time.time()
    # print('query_embed:',query_embedding_end-embed_end)

    # similarity_scores = cosine_similarity([query_embedding], embeddings_np)
    # indices = np.argsort(-similarity_scores)

    # print('search_time:',time.time()-query_embedding_end)
    


    return embeddings_np

if __name__ == '__main__':
    result = main()

