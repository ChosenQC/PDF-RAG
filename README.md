# RAG Workflow Demo

本项目演示了基于 Haystack、FlagEmbedding、HNSWLib 和 BM25 的文档检索与向量化流程。

## 目录结构

- `RAG_workflow/parse.py`：PDF 文档解析与切分，生成 JSON 格式的文档内容。
- `RAG_workflow/embedding.py`：对文档内容进行向量化，生成 embedding vector base。
- `RAG_workflow/HNSW_retrieve.py`：基于 HNSW 和 BM25 进行hybrid检索与召回。

## 环境依赖

请先安装依赖：

```bash
pip install -r requirements.txt
```

## 使用流程

1. **PDF 文档解析**
   - 修改 `parse.py` 中 `PATH_TO_YOUR_PDF_DIRECTORY` 为你的 PDF 文件夹路径。
   - 修改输出 JSON 路径（`PATH_TO_YOUR_JSON`）。
   - 运行：
     ```bash
     python RAG_workflow/parse.py
     ```
   - 生成的 JSON 文件将用于后续embedding。

2. **JSON文件向量化**
   - 修改 `embedding.py` 中 JSON 路径（`PATH_TO_YOUR_JSON.json`）和输出 embedding 路径（`PATH_TO_YOUR_EMBEDDING.npy`）。
   - 运行：
     ```bash
     python RAG_workflow/embedding.py
     ```

3. **检索与召回**
   - 修改 `HNSW_retrieve.py` 中 embedding 路径和 JSON 路径。
   - 运行：
     ```bash
     python RAG_workflow/HNSW_retrieve.py
     ```
   - 控制台会输出 HNSW 和 BM25 的构建与检索时间，以及合并后的检索结果。
   - 按照description调整HNSW 和 BM25的参数以获取需要的结果。
   - hnswlib.Index()中space='l2' for Squared L2,'ip' for Inner product and 'cosine' for Cosine similarity.

## 依赖包说明

- numpy
- scikit-learn
- hnswlib
- rank_bm25
- FlagEmbedding
- haystack
- haystack-integrations



---

如有问题欢迎提 issue！
