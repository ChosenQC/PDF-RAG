import json
import numpy as np
from FlagEmbedding import FlagAutoModel
from FlagEmbedding import FlagModel
import time 
from sklearn.metrics.pairwise import cosine_similarity
import os

def main():
#    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    start_time = time.time()
    # 加载模型
    model = FlagAutoModel.from_finetuned(
        'BAAI/bge-base-en-v1.5',
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        #  devices='cpu',
         use_fp16=True)

    model_end = time.time()
    global model_loading_time 
    model_loading_time = model_end - start_time
    print(model_loading_time)
    # 读取新的JSON数据文件
    with open('./datasets/documents_1000.json', 'r', encoding='utf-8') as file:
       new_data = json.load(file)
   
   # 检查有多少个数据
    print(len(new_data))
    embed_start = time.time()
    # 只提取content字段
    to_embed = []
    for document in new_data:
       content = document.get('content', '')  # 直接获取content字段
       to_embed.append(content.strip())  # 去除前后空白后添加到列表
    
    # 生成嵌入向量数组
    embeddings_np = np.array(model.encode(to_embed))

    # 直接导入嵌入向量数组
    # embeddings_np=np.load('./10000_gpu.npy')
    
    print('input_shape:',embeddings_np.shape)
    
    np.save('1000_gpu.npy', embeddings_np)
    embed_end = time.time()
    global embedding_time 
    embedding_time = embed_end - embed_start
    print('embedding_time:',embedding_time)

    query='Physics Paper'
    query_embedding=np.array(model.encode(query))
    query_embedding_end=time.time()
    print('query_embed:',query_embedding_end-embed_end)

    similarity_scores = cosine_similarity([query_embedding], embeddings_np)
    
    print('find_time:',time.time()-query_embedding_end)
    indices = np.argsort(-similarity_scores)
    end_time=time.time()
    print('query_and_found time:',end_time-embed_end)
# 返回NumPy数组结果
    return embeddings_np

if __name__ == '__main__':
    result = main()

    # 验证并输出结果
    print("输出嵌入向量数组:")
    print(result)
    print("\n数组详细信息:")
    print(f"类型: {type(result)}")
    print(f"形状: {result.shape} (文档数: {result.shape[0]}, 嵌入维度: {result.shape[1]})")
