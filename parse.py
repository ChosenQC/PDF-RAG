from pathlib import Path
import time
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document, Pipeline
import json

start_time=time.time()
document_store = InMemoryDocumentStore()

## Create pipeline components
converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="sentence", split_length=1)
writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

## Add components to the pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("cleaner", cleaner)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("writer", writer)

## Connect the components to each other
indexing_pipeline.connect("converter", "cleaner")
indexing_pipeline.connect("cleaner", "splitter")
indexing_pipeline.connect("splitter", "writer")
inni_time=time.time()
print('init_time=',inni_time-start_time)
papers_dir = Path("PATH_TO_YOUR_PDF_DIRECTORY")  # Replace with your actual directory path
pdf_files = list(papers_dir.glob("*.pdf"))
print(len(pdf_files))
i=0
for pdf_file in pdf_files:    
    try:
        indexing_pipeline.run({
            "converter": {
                "sources": [pdf_file]
            }
        })
        i=i+1
    except Exception as e:
        print(f"Missing {pdf_file.name}: {str(e)}")
index_time=time.time()
print('index_time_=',index_time-inni_time)
all_documents = document_store.filter_documents()
docs_list = [doc.to_dict() for doc in all_documents]

# 存储为JSON文件
with open("PATH_TO_YOUR_JSON", "w", encoding="utf-8") as f:
    json.dump(docs_list, f, ensure_ascii=False, indent=2)

