import os
import time
import argparse
import pandas as pd

from langchain import hub
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    vectorstore_path,
    embedding_model,
    reranker_model,
    llm_model,
    score_threshold,
    initial_k,
    rerank_top_n,
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    db = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embeddings,
    )

    base_retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": score_threshold,
            "k": initial_k,
        },
    )

    rerank_model = HuggingFaceCrossEncoder(model_name=reranker_model)
    compressor = CrossEncoderReranker(
        model=rerank_model,
        top_n=rerank_top_n,
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
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
        {
            "context": compression_retriever,
            "question": RunnablePassthrough(),
        }
    ).assign(answer=rag_chain_from_docs)

    return rag_chain


def run_batch_inference(rag_chain, input_csv, output_csv, encoding="utf-8-sig"):
    df = pd.read_csv(input_csv, encoding=encoding)

    if "user_input" not in df.columns:
        raise ValueError("Input CSV must contain a 'user_input' column.")

    questions = df["user_input"]

    answers = []
    contexts = []
    response_times = []

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

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding=encoding)

    print(f"[SUCCESS] Results saved to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Rerank-RAG inference."
    )

    parser.add_argument(
        "--vectorstore",
        required=True,
        help="Path to Chroma vectorstore.",
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to input question CSV. It must contain a 'user_input' column.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to output CSV file.",
    )

    parser.add_argument(
        "--embedding_model",
        default="intfloat/multilingual-e5-small",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--reranker_model",
        default="BAAI/bge-reranker-base",
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument(
        "--llm_model",
        default="llama3.1:8b",
        help="Ollama LLM model name.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Similarity score threshold for initial retrieval.",
    )
    parser.add_argument(
        "--initial_k",
        type=int,
        default=10,
        help="Number of documents retrieved before reranking.",
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=10,
        help="Number of documents kept after reranking.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rag_chain = build_rag_chain(
        vectorstore_path=args.vectorstore,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        llm_model=args.llm_model,
        score_threshold=args.score_threshold,
        initial_k=args.initial_k,
        rerank_top_n=args.rerank_top_n,
    )

    run_batch_inference(
        rag_chain=rag_chain,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
