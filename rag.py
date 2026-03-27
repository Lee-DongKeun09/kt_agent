from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_DIR = "data/pdf"
VECTORSTORE_PATH = "vectorstore"

def load_pdfs(pdf_dir: str):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(pdf_dir, filename))
            docs.extend(loader.load())
            print(f"  로드됨: {filename} ({len(loader.load())}페이지)")
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  총 청크 수: {len(chunks)}")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"  벡터스토어 저장 완료: {VECTORSTORE_PATH}/")
    return vectorstore

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""아래 문서 내용을 바탕으로 질문에 답하세요.
문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 답하세요.

문서 내용:
{context}

질문: {question}
답변:"""
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def summarize(vectorstore):
    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)
    docs = vectorstore.similarity_search("문서의 주요 내용 핵심 요약", k=8)
    context = "\n\n".join(d.page_content for d in docs)
    response = llm.invoke(
        f"아래 문서 내용을 3~5줄로 핵심만 요약해주세요.\n\n{context}"
    )
    return response.content

def main():
    print("=== RAG 시스템 ===\n")

    # 벡터스토어 로드 or 새로 빌드
    if os.path.exists(VECTORSTORE_PATH):
        print("기존 벡터스토어 로드 중...")
        vectorstore = load_vectorstore()
    else:
        print("PDF 로드 중...")
        docs = load_pdfs(PDF_DIR)
        print(f"\n벡터스토어 구축 중...")
        vectorstore = build_vectorstore(docs)

    qa_chain = build_qa_chain(vectorstore)
    print("\n준비 완료! 명령어: 'q <질문>' | 'summary' | 'rebuild' | 'exit'\n")

    while True:
        user_input = input("입력> ").strip()
        if not user_input:
            continue
        elif user_input == "exit":
            break
        elif user_input == "summary":
            print("\n[요약 중...]\n")
            print(summarize(vectorstore))
            print()
        elif user_input == "rebuild":
            print("\nPDF 다시 로드 중...")
            docs = load_pdfs(PDF_DIR)
            print("벡터스토어 재구축 중...")
            vectorstore = build_vectorstore(docs)
            qa_chain = build_qa_chain(vectorstore)
            print("완료!\n")
        elif user_input.startswith("q "):
            question = user_input[2:].strip()
            print("\n[답변 중...]\n")
            result = qa_chain.invoke(question)
            print(result)
            print()
        else:
            print("명령어: 'q <질문>' | 'summary' | 'rebuild' | 'exit'\n")

if __name__ == "__main__":
    main()
