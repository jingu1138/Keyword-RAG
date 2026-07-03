import os
import argparse

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import TextLoader


def load_markdown_document(input_path, encoding="utf-8-sig"):
    loader = TextLoader(input_path, encoding=encoding)
    docs = loader.load()
    return docs[0].page_content


def split_markdown_by_header(text):
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1")],
        strip_headers=True,
    )

    return markdown_splitter.split_text(text)


def chunk_documents(md_docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", " "],
    )

    final_docs = []

    for doc in md_docs:
        sub_chunks = text_splitter.split_text(doc.page_content)

        for chunk in sub_chunks:
            final_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata,
                )
            )

    return final_docs


def build_vectorstore(documents, output_path, embedding_model):
    os.makedirs(output_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=output_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Chroma vectorstore from a Markdown technical document."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input Markdown or text file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output Chroma vectorstore directory.",
    )
    parser.add_argument(
        "--embedding_model",
        default="intfloat/multilingual-e5-small",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Maximum chunk size.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Chunk overlap size.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading document...")
    text = load_markdown_document(args.input)

    print("[INFO] Splitting document by Markdown headers...")
    md_docs = split_markdown_by_header(text)
    print(f"[INFO] Header-level sections: {len(md_docs)}")

    print("[INFO] Creating text chunks...")
    final_docs = chunk_documents(
        md_docs=md_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"[INFO] Final chunks: {len(final_docs)}")

    print("[INFO] Building Chroma vectorstore...")
    build_vectorstore(
        documents=final_docs,
        output_path=args.output,
        embedding_model=args.embedding_model,
    )

    print(f"[SUCCESS] Vectorstore saved to: {args.output}")


if __name__ == "__main__":
    main()
