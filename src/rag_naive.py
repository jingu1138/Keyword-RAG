import os
import time
import argparse
import pandas as pd

from langchain import hub
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore_path, embedding_model, llm_model, score_threshold, k):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    db = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": score_threshold,
            "k": k,
        },
    )

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model=llm_model, num_gpu=1, temperature=0)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain


def run_batch_inference(rag_chain, input_csv, output_path, encoding="utf-8-sig"):
    df = pd.read_csv(input_csv, encoding=encoding)

    if "user_input" not in df.columns:
        raise ValueError("Input CSV must contain a 'user_input' column.")

    answers = []
    contexts = []
    response_times = []

    questions = df["user_input"]

    print(f"[INFO] Start batch inference: {len(questions)} questions")

    for idx, question in enumerate(questions, 1):
        start_time = time.time()

        result = rag_chain.invoke(question)

        elapsed_time = time.time() - start_time

        answers.append(result["answer"])
        contexts.append([doc.page_content for doc in result["context"]])
        response_times.append(round(elapsed_time, 3))

        print(f"[{idx}/{len(questions)}] Done - {elapsed_time:.2f}s")

    df["answer"] = answers
    df["context"] = contexts
    df["response_time"] = response_times

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)

    print(f"[SUCCESS] Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG inference.")

    parser.add_argument("--vectorstore", required=True, help="Path to Chroma vectorstore")
    parser.add_argument("--input_csv", required=True, help="Path to input question CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output result CSV")

    parser.add_argument("--embedding_model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--llm_model", default="llama3.1:8b")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    rag_chain = build_rag_chain(
        vectorstore_path=args.vectorstore,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        score_threshold=args.score_threshold,
        k=args.k,
    )

    run_batch_inference(
        rag_chain=rag_chain,
        input_csv=args.input_csv,
        output_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
