from flask import Flask, request, jsonify
import argparse
import os

from langchain.vectorstores import Chroma
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)


def step(args):
    global embedding, vectordb, llm, conversational_template
    print("start step llm ....")
    # 初始化 大模型相关 信息
    file_path = args.file
    prompt = args.prompt

    embedding = OpenAIEmbeddings(openai_api_key="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")
    llm = ChatOpenAI(openai_api_key="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")

    # 根据文件 生成 本地索引
    persist_directory = 'text/'
    if os.path.exists(persist_directory):
        print("load local embedding ...")
        vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    else:
        # embedding
        print("load local file...")
        loader = Docx2txtLoader(file_path=file_path)
        doc = loader.load()
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_spliter.split_documents(doc)
        print("file to local embedding ...")
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)

    conversational_template = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever(),
                                                                    return_source_documents=True,
                                                                    return_generated_question=True)

    print("end step llm ....")


@app.route("/chat", methods=["POST"])
def chat():

    chat_data = [{"input": item['input'], "output": item['output']} for item in request.json]

    # chat history
    chat_history = [(item['input'], item['output']) for item in chat_data[:-1]]
    print(f"chat history {chat_history}")
    # new question
    question_data = chat_data[-1]
    print(f"chat question {question_data}")

    result = conversational_template({"question": question_data['input'], "chat_history": chat_history})
    print(result)
    answer = result['answer']

    return jsonify({"input": question_data['input'], "output": answer})


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', "*")
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="dsl template file path")
    parser.add_argument('--prompt', type=str, default="prompt", help="code work space")
    args = parser.parse_args()

    step(args)

    app.run(port=8080)
