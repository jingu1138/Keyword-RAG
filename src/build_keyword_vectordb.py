import os
import json
import time
import argparse
import torch

from adaptkeybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
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


def filter_similar_keywords(keywords, sim_model, threshold=0.7):
    if len(keywords) < 2:
        return keywords

    embeddings = sim_model.encode(keywords, convert_to_tensor=True)

    keep = [keywords[0]]
    keep_indices = [0]

    for i in range(1, len(keywords)):
        is_similar = False

        for keep_idx in keep_indices:
            similarity = util.cos_sim(embeddings[i], embeddings[keep_idx]).item()

            if similarity >= threshold:
                is_similar = True
                break

        if not is_similar:
            keep.append(keywords[i])
            keep_indices.append(i)

    return keep


def extract_keywords_from_sections(
    md_docs,
    keyword_model,
    sim_model,
    sim_threshold=0.7,
    adapt_len_limit=500,
    top_n=10,
    ngram_range=(1, 1),
):
    keyworded_sections = []

    adapt_time_total = 0.0
    adapt_count = 0

    for doc in md_docs:
        content = doc.page_content
        metadata = doc.metadata

        keywords = []

        if len(content) > adapt_len_limit:
            adapt_count += 1
            adapt_start = time.perf_counter()

            try:
                keyword_model.pre_train(
                    [content],
                    [["supervised", "unsupervised"]],
                    lr=1e-3,
                )
                keyword_model.zeroshot_pre_train(
                    ["supervised", "unsupervised"],
                    adaptive_thr=0.15,
                )
            except Exception as error:
                print(f"[WARN] AdaptKeyBERT adaptation skipped: {error}")

            adapt_time_total += time.perf_counter() - adapt_start

        extracted = keyword_model.extract_keywords(
            content,
            top_n=top_n,
            keyphrase_ngram_range=ngram_range,
        )

        if extracted:
            keyword_candidates = [keyword for keyword, _ in extracted]
            keywords = filter_similar_keywords(
                keyword_candidates,
                sim_model=sim_model,
                threshold=sim_threshold,
            )

        keyworded_sections.append(
            {
                "content": content,
                "keywords": keywords,
                "metadata": metadata,
            }
        )

    return keyworded_sections, adapt_time_total, adapt_count


def save_keywords_json(keyworded_sections, output_json):
    output_dir = os.path.dirname(output_json)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(keyworded_sections, file, ensure_ascii=False, indent=4)


def create_keyword_augmented_chunks(
    keyworded_sections,
    chunk_size=500,
    chunk_overlap=50,
    adapt_len_limit=500,
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", " "],
    )

    chunked_documents = []

    for section in keyworded_sections:
        content = section["content"]
        metadata = section["metadata"]
        keywords = section.get("keywords", [])

        if not content:
            continue

        keyword_header = ""
        if len(content) > adapt_len_limit and keywords:
            keyword_header = f"[KEYWORDS: {', '.join(keywords)}]\n"

        sub_chunks = text_splitter.split_text(content)

        for chunk in sub_chunks:
            chunk = chunk.strip()

            if not chunk:
                continue

            page_content = keyword_header + chunk if keyword_header else chunk

            chunked_documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

    return chunked_documents


def build_vectorstore(documents, output_path, embedding_model):
    os.makedirs(output_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=output_path,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a keyword-augmented Chroma vectorstore for Keyword-RAG."
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
        "--output_json",
        default=None,
        help="Path to save extracted keywords as JSON.",
    )

    parser.add_argument(
        "--embedding_model",
        default="intfloat/multilingual-e5-small",
        help="Embedding model for vectorstore indexing.",
    )
    parser.add_argument(
        "--similarity_model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for keyword similarity filtering.",
    )

    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for removing redundant keywords.",
    )
    parser.add_argument(
        "--adapt_len_limit",
        type=int,
        default=500,
        help="Minimum section length for AdaptKeyBERT domain adaptation.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Maximum number of extracted keywords per section.",
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading document...")
    text = load_markdown_document(args.input)

    print("[INFO] Splitting document by Markdown headers...")
    md_docs = split_markdown_by_header(text)
    print(f"[INFO] Header-level sections: {len(md_docs)}")

    print("[INFO] Loading keyword extraction models...")
    keyword_model = KeyBERT(domain_adapt=True, zero_adapt=True)
    sim_model = SentenceTransformer(args.similarity_model, device=device)

    print("[INFO] Extracting representative keywords...")
    start_time = time.perf_counter()

    keyworded_sections, adapt_time_total, adapt_count = extract_keywords_from_sections(
        md_docs=md_docs,
        keyword_model=keyword_model,
        sim_model=sim_model,
        sim_threshold=args.sim_threshold,
        adapt_len_limit=args.adapt_len_limit,
        top_n=args.top_n,
        ngram_range=(1, 1),
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"[INFO] Keyword extraction time: {elapsed_time:.2f}s")
    print(
        f"[INFO] Adaptation time: {adapt_time_total:.2f}s "
        f"(avg: {(adapt_time_total / adapt_count if adapt_count else 0):.2f}s, n={adapt_count})"
    )

    if args.output_json:
        print("[INFO] Saving extracted keywords...")
        save_keywords_json(keyworded_sections, args.output_json)
        print(f"[INFO] Keywords saved to: {args.output_json}")

    print("[INFO] Creating keyword-augmented chunks...")
    chunked_documents = create_keyword_augmented_chunks(
        keyworded_sections=keyworded_sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        adapt_len_limit=args.adapt_len_limit,
    )
    print(f"[INFO] Final chunks: {len(chunked_documents)}")

    print("[INFO] Building keyword-augmented Chroma vectorstore...")
    build_vectorstore(
        documents=chunked_documents,
        output_path=args.output,
        embedding_model=args.embedding_model,
    )

    print(f"[SUCCESS] Keyword vectorstore saved to: {args.output}")


if __name__ == "__main__":
    main()
