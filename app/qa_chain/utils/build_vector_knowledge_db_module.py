# -*- coding: utf-8 -*-

# ***************************************************
# * File        : build_vectordb.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-22
# * Version     : 0.1.102200
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from dotenv import load_dotenv, find_dotenv

# OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 智谱 Embedding
from embedding_api.zhipuai_embedding import ZhipuAIEmbeddings

from app.qa_chain.utils.data_provider.pdf_loader import load_pdf
from app.qa_chain.utils.data_provider.markdown_loader import load_markdown
from app.qa_chain.utils.data_processor.pdf_process import process_pdf
from app.qa_chain.utils.data_processor.markdown_process import process_markdown
from app.qa_chain.utils.doc_spliter import recur_split_doc
from app.qa_chain.utils.chroma_utils import chroma_save

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
# 如果需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'


def get_docs(knowledge_lib_path: str = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib", 
             ):
    """
    获取、处理、分割知识库文本数据
    """
    # ------------------------------
    # data load
    # ------------------------------
    # 获取 folder_path 下所有文件路径，储存在 file_paths 里
    file_paths = []
    # folder_path = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/knowledge_lib"
    folder_path = knowledge_lib_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths[:3])
    # 遍历文件路径并把实例化的 loader 存放在 loaders 里
    loaders = []
    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            pages = load_pdf(doc_path = file_path)
            processed_pages = process_pdf(pdf_page = pages)
            loaders.append(processed_pages)
        elif file_type == 'md':
            pages = load_markdown(doc_path = file_path)
            processed_pages = process_markdown(md_page = pages)
            loaders.append(processed_pages)
    # 下载文件并存储到 text
    texts = []
    for loader in loaders: 
        texts.extend(loader)
    # 查看数据
    # text = texts[1]
    # print(
    #     f"每一个元素的类型：{type(text)}.", 
    #     f"该文档的描述性数据：{text.metadata}", 
    #     f"查看该文档的内容:\n{text.page_content[0:]}", 
    #     sep="\n------\n"
    # )
    # ------------------------------
    # doc split
    # ------------------------------
    split_docs = recur_split_doc(pages = texts,chunk_size = 500, overlap_size = 50)
    
    return split_docs


def build_vectordb(persist_directory: str = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/vector_db/chroma"):
    # ------------------------------
    # 获取、处理、分割知识库文本数据
    # ------------------------------
    split_docs = get_docs()
    # ------------------------------
    # 定义 Embeddings
    # ------------------------------
    # embedding = OpenAIEmbeddings() 
    # embedding = QianfanEmbeddingsEndpoint()
    embedding = ZhipuAIEmbeddings()
    # ------------------------------
    # 构建向量数据库
    # ------------------------------
    # 定义持久化路径
    # persist_directory = "E:/projects/llms_proj/llm_proj/app/qa_chain/database/vector_db/chroma"
    # 存储向量数据
    vectordb = chroma_save(split_docs, embedding, persist_directory)

    return vectordb




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
