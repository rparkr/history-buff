import marimo

__generated_with = "0.9.9"
app = marimo.App()


@app.cell
def __():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore

    # from langchain_huggingface import HuggingFaceEmbeddings  # requires a ton of additional software
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings,
    )  # requires only sentence_transformers

    # See also BgeEmbeddings for BGE and Nomic models, including the ability to set a prompt:
    # "query instruction" and "embed instruction" to prefix each document
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    # from langchain.embeddings import HuggingFaceBgeEmbeddings
    return (
        AsyncHtmlLoader,
        Document,
        FAISS,
        HuggingFaceBgeEmbeddings,
        HuggingFaceEmbeddings,
        VectorStore,
        WebBaseLoader,
    )


@app.cell
def __():
    import os

    # Set environment variables for web scraping
    os.environ["USER_AGENT"] = (
        "history-buff: semantic search over web browsing history. https://github.com/rparkr/history-buff"
    )

    # from langchain.document_loaders import WebBaseLoader
    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_community.document_transformers import MarkdownifyTransformer
    from langchain_text_splitters import MarkdownTextSplitter
    return AsyncHtmlLoader, MarkdownTextSplitter, MarkdownifyTransformer, os


@app.cell
def __(AsyncHtmlLoader, MarkdownTextSplitter, MarkdownifyTransformer):
    urls = [
        "https://churchofjesuschristtemples.org/payson-utah-temple/",
        "https://www.deeplearning.ai/the-batch/issue-270/",
        "https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/",
        "https://scikit-learn.org/stable/modules/clustering.html",
        "https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.explain.html",
        "https://docs.marimo.io/guides/plotting.html",
    ]

    doc_loader = AsyncHtmlLoader(urls)
    md_transformer = MarkdownifyTransformer(
        strip=["img", "a", "b", "em"],
        bullets="-",  # default: "*+-", where each successive level uses a different bullet
        code_language="python",  # set the default language for code blocks
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
        # For more options, see: https://github.com/matthewwithanm/python-markdownify/tree/develop
    )
    md_text_splitter = MarkdownTextSplitter()
    raw_docs = doc_loader.load()
    # For async variants, use:
    # Return list of Documents: await doc_loader.aload()
    # Return list of str: await doc_loader.fetch_all()
    return doc_loader, md_text_splitter, md_transformer, raw_docs, urls


@app.cell
def __(md_text_splitter, md_transformer, raw_docs):
    md_docs = md_transformer.transform_documents(raw_docs)
    # Async version:
    # md_docs = await md_transformer.atransform_documents(raw_docs)

    chunked_docs = md_text_splitter.transform_documents(md_docs)
    # Async version:
    # chunked_docs = await md_text_splitter.atransform_documents(md_docs)
    return chunked_docs, md_docs


@app.cell
def __(md_docs):
    import re

    {
        doc.metadata["title"]: len(re.split("\s", doc.page_content))
        for doc in md_docs
    }
    return (re,)


@app.cell
def __(chunked_docs):
    chunked_docs[3].metadata
    return


@app.cell
def __(chunked_docs):
    num_chunks = {}
    for doc in chunked_docs:
        if doc.metadata["source"] not in num_chunks:
            num_chunks[doc.metadata["source"]] = 1
        else:
            num_chunks[doc.metadata["source"]] += 1

    num_chunks
    return doc, num_chunks


app._unparsable_cell(
    r"""
    embedding_models = [
        \"Alibaba-NLP/gte-base-en-v1.5\",  # 137M params; 8192 context; https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5
        \"nomic-ai/nomic-embed-text-v1.5\",  # 137M params; 8192 context; a separate image encoder is available shares embedding space; https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        \"avsolatorio/GIST-small-Embedding-v0\",  # 33M params; 512 context; https://huggingface.co/avsolatorio/GIST-small-Embedding-v0
        \"BAAI/bge-small-en-v1.5\";  # 33M params; 512 context; https://huggingface.co/BAAI/bge-small-en-v1.5; prompt for queries: \"Represent this sentence for searching relevant passages: \" (see: https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)
    ]
    model_name = \"Alibaba-NLP/gte-base-en-v1.5\"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    """,
    name="__"
)


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
