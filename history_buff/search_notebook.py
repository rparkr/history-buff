import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Semantic search

        In this notebook, I experiment with setting up semantic search using `LangChain`.

        Steps:

        1. Create `LangChain` `Document`s from a list of URLs
        2. Add metadata to the URLs from information in the browser history database
        3. Create (or load) a vector store (I use `faiss` - Facebook AI Similarity Search)
        4. Process the documents by converting the HTML into Markdown (which is much cleaner for the embedding model to process and also reduces the token count)
        5. Split documents into _chunks_ of tokens that fit within the embedding model's max context length
        6. Embed the document chunks, creating a vector representation for each chunk
        7. Add the document embeddings to the vector store

        When a user enters a search term, the vector store computes a similarity search between the user's query and the document chunks, and returns the webpage (URL) corresponding to the highest similarity score.

        ## References

        I studied the following resources while experimenting with this approach:

        - LangChain docs: [Faiss](https://python.langchain.com/docs/integrations/vectorstores/faiss/) and [Faiss async](https://python.langchain.com/docs/integrations/vectorstores/faiss_async/). I use the asynchronous version as the vector database in this similarity search application.
            - [FAISS API reference](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html)
            - See also: [langchain_core.vectorstores.VectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html), which is a vector store implementation that does not require additional dependencies (like FAISS).
        - [LangChain conceptual guide](https://python.langchain.com/docs/concepts), helpful documentation on many concepts like document loaders, vector stores, embedding models, and everything else in the LangChain ecosystem
            - For example, here's [a how-to guide on the basics of vector stores](https://python.langchain.com/docs/how_to/vectorstores/)
        - API reference: [AsyncHtmlLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.async_html.AsyncHtmlLoader.html): asynchronously loads text from a provided list of URLs, either in lazy mode (which I believe means that computation is delayed until requested) or in eager mode. It can loads the raw HTML from web pages into Documents, which are then processed through transformations like MarkdownifyTransformer
        - [`markdownify`](https://github.com/matthewwithanm/python-markdownify/tree/develop): an MIT-licensed pure-Python package for converting HTML text into Markdown. Markdown is much easier to read for both humans and LLMs, and it focuses on the content, enabling an embedding model to capture the semantic meaning of the page without being distracted by the HTML tags used for structuring the document.
        - LangChain API for [MarkdownifyTransformer](https://python.langchain.com/api_reference/community/document_transformers/langchain_community.document_transformers.markdownify.MarkdownifyTransformer.html), a document transformer that implements `markdownify` and comes with handy asynchronous functions. It takes a list of `Document`s with raw HTML strings as their `page_content` and transforms the `page_content` string into Markdown. I found that the combination of `AsyncHtmlLoader` and `MarkdownifyTransformer` was more effective than the `WebBaseLoader` (which removes the HTML tags but does not format the page as nicely as the Markdown version)
        - LangChain API reference: [`HuggingFaceBgeEmbeddings`](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceBgeEmbeddings.html): this is an alternative to `langchain_huggingface.HuggingFaceEmbeddings` and does not require as many dependencies (only `sentence-tranformers`, rather than the `PyTorch` dependencies required by `langchain-huggingface`). This class is used to load a Hugging Face model and create document embeddings
        - LangChain API reference: [`FastEmbedEmbeddings`](https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html), an alternative to `HuggingFaceBgeEmbeddings` that uses [fastembed](https://github.com/qdrant/fastembed/), which claims to run faster (it uses ONNX runtime) and require fewer dependencies than Hugging Face embeddings. It does not offer as many models, but the ones it includes are high-performing models based on the MTEB leaderboard. It also has an interface for creating image embeddings (which might be similar to Nomic, where the text embedding model and image embedding model share the same embedding space; but for Nomic the image embedding model has a more restrictive license).
        - [Hugging Face Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference), a super-fast embeddings serving framework written in Rust and run as a lightweight Docker container. It could be helpful to generate embeddings via an API call to the service, rather than through LangChain's classes.
        - The [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard), where you can learn about different embeddings models available. I filtered based on license (permissive), model size (smaller), and context length (8192), then selected some of the highest-performing models as of October 2024.
            - I believe that the models must be supported on `sentence-transformers` to use with the `HuggingFaceBgeEmbeddings` class in LangChain, but I'm not 100% sure (I haven't yet experimented with a model that is not available with `sentence-transformers`)
        - Embeddings models:
            - [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5): 137M params, 8192 context length (thanks to RoPE positional encodings).
            - [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5): 137M params, 8192 context length. A separate image encoder is available with shared embedding space, but the image encoder has a non-commercial license (the text encoder is licensed under Apache 2.0). The query prompt is: `"search_query: "` and the embedding prompt is: `"search_document: "`.
            - [avsolatorio/GIST-small-Embedding-v0](https://huggingface.co/avsolatorio/GIST-small-Embedding-v0): 33M params, 512 context length. Very high performance on MTEB for models with <100M parameters.
            - [BAAI/bge-small-en-v1](https://huggingface.co/BAAI/bge-small-en-v1.5#using-sentence-transformers): 33M params, 512 context. The "Using sentence transformers" section on this page is where I learned about some of the configuration options to pass to `HuggingFaceBgeEmbeddings` (e.g., normalize) and also where I learned about the query and embed prompts used by BGE (and Nomic) models. The query prompt for this model is: `"Represent this sentence for searching relevant passages: "`. The BGE family of models is sometimes referred to as FlagEmbedding models based on the software package BAAI build to serve those models. You can learn more about the available models from BAAI [here](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)).
        - [TextSplitters API reference](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/). I use the [MarkdownTextSplitter](https://python.langchain.com/api_reference/text_splitters/markdown/langchain_text_splitters.markdown.MarkdownTextSplitter.html) for _chunking_ documents into sizes that fit within the model's context length. In general, the MarkdownTextSplitter performs very well -- even if using a smaller, character-based context length (the [ChunkViz visualizer](https://chunkviz.up.railway.app/) demonstrates how the MarkdownTextSplitter breaks the text into chunks). The version of MarkdownTextSplitter I use is based on Markdown headers rather than individual characters, which means it groups together sections of the webpage based on the page's natural structure (or at least, the structure it is defined with based on its HTML).
        """
    )
    return


@app.cell
def __():
    import asyncio
    import os
    import uuid

    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_community.document_transformers import MarkdownifyTransformer
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
    from langchain_text_splitters import MarkdownTextSplitter

    # Set environment variables for web scraping
    os.environ["USER_AGENT"] = (
        "history-buff: semantic search over web browsing history. https://github.com/rparkr/history-buff"
    )
    return (
        AsyncHtmlLoader,
        Document,
        FAISS,
        HuggingFaceEmbeddings,
        MarkdownTextSplitter,
        MarkdownifyTransformer,
        VectorStore,
        asyncio,
        os,
        uuid,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Token counter

        In this section, I implement a class that shows the maximum context length for a given embedding model, which can improve the accuracy in chunking (splitting) the Markdown documents into chunks that fit within the embedding model's context widow.

        Note that using this class is optional. When first used to determine a model's max input length, the `AutoTokenizer` class from the `transformers` library is loaded, which requires about 300 MB of RAM.
        """
    )
    return


@app.cell
def __():
    class TokenCounter:
        def __init__(self, model_name: str, verbose: bool = False, **kwargs):
            """
            Count the number of tokens in a text and determine a model's
            max input context length (in tokens) using the model's tokenizer.
            """
            # Here's an alternative way to save all the inputs
            # as instance attributes:
            # self.__dict__.update(locals())
            self.model_name = model_name
            self.verbose = verbose
            self.kwargs = kwargs
            self.tokenizer = None
            self.max_length = None

        def load_tokenizer(self):
            """
            Load the tokenizer to determine token counts and max context length.

            With `verbose=True`, this will print the max context length of the
            model. You can always access the model's context length with the
            max_length attribute after you have loaded the tokenizer.

            Running this will import AutoTokenizer from the transformers library,
            which uses about 300 MB of RAM.
            """
            # Lazy-load this module since it requires ~300 MB of RAM
            from transformers import AutoTokenizer

            # Load the tokenizer to determine the chunk size allowed by the model
            # and the method for determining the number of tokens in a document
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Sometimes the tokenizer.model_max_length attribute returns a number
            # that is larger than the model's actual max sequence length, so I
            # recommend capping the maximum at 8192, which was the maximum length
            # among leading embedding models as of October 2024.
            self.max_length = self.tokenizer.model_max_length
            if self.verbose:
                print(
                    "Tokenizer loaded. The model's max input length is "
                    f"{self.max_length} tokens."
                )
            return self

        def count_tokens(self, text: str) -> int:
            """
            Return the count of tokens for a given string of text.
            """
            if not self.tokenizer:
                print("Tokenizer not loaded. Run load_tokenizer() first.")
            return len(self.tokenizer.tokenize(text))

        def tokenize(self, text: str) -> list[str]:
            """
            Return a list of tokens for a given string of text.
            """
            if not self.tokenizer:
                print("Tokenizer not loaded. Run load_tokenizer() first.")
            return self.tokenizer.tokenize(text)

        def __repr__(self):
            return f"TokenCounter(model_name='{self.model_name}')"
    return (TokenCounter,)


@app.cell
def __(TokenCounter):
    def test_token_counter(models: list = []):
        for model_name in models:
            print("=" * 80, f"Model: {model_name}".center(80), "=" * 80, sep="\n")
            token_counter = TokenCounter(model_name=model_name, verbose=True)
            token_counter.load_tokenizer()
            sample_text = """
            # This is a markdown header

            And this is text in a section.

            ```python
            # Here is a code block

            def say_hi():
                print('hi')
            ```
            """
            print(sample_text)
            print("Num tokens:", token_counter.count_tokens(sample_text))
            print("Tokens:", token_counter.tokenize(sample_text))


    test_token_counter(
        models=[
            "Alibaba-NLP/gte-base-en-v1.5",
            "nomic-ai/nomic-embed-text-v1.5",
            "avsolatorio/GIST-small-Embedding-v0",
            "BAAI/bge-small-en-v1.5",
        ]
    )
    return (test_token_counter,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Create and prepare documents

        In this section, I load `Documents` from a list of URLs, add metadata, and process the documents by converting them to Markdown and then splitting them into chunks that fit within the embedding model's maximum context length.
        """
    )
    return


@app.cell
def __(Document, uuid):
    async def add_metadata_to_document(
        doc,
        metadata: dict,
        add_id: bool = True,
        id_key: str = "id",
        overwrite_exising_id: bool = False,
    ) -> Document:
        """
        Add metadata to a LangChain Document.

        Returns the Document with updated metadata.

        If add_id is True (default), the Document's `id` attribute
        will be updated using the `id_key` in metadata, if available,
        otherwise a new ID will be generated for the Document.

        If overwrite_exising_id is False and the Document already
        had its id attribute set, then the existing id will not
        be overwritten.

        Notes
        ------
        For the id parameters:
        In the case of semantic search over browsing history, it is
        helpful to connect the id of a Document to the id from
        the browser's history database (which is just an integer
        starting from 1 and incrementing by 1 for each time a page
        is viewed for the first time).
        """
        doc.metadata.update(metadata)
        if add_id and (not doc.id or overwrite_exising_id):
            if id_key in metadata:
                doc.id = metadata[id_key]
            else:
                doc.id = str(uuid.uuid4())
        return doc
    return (add_metadata_to_document,)


@app.cell
def __(AsyncHtmlLoader, Document, add_metadata_to_document, asyncio):
    async def create_documents_from_urls(
        urls: list[str], metadata: dict[str, list[str]] | None = None, **kwargs
    ) -> list[Document]:
        """
        Load all pages' HTML text into Documents along with metadata.

        Parameters
        ----------
        urls: list of str
            The URLs whose HTML text will be used for Documents.
        metadata: dict
            The metadata to be added to each document. The keys are
            the metadata keys and the values are lists with the same
            number of elements as `urls`.
        **kwargs
            You can also provided metadata as keyword arguments, where
            the parameter names become the metadata fields and the
            parameter values are lists with the same number of elements
            as `urls`.
            If the same key exists in both `metadata` and `kwargs`, the
            value in `kwargs` will overwrite the value in `metadata`.

        Returns
        -------
        List of langchain Document objects, where the `page_content` field
        holds the HTML text of each URL.
        """
        # Validate that metadata corresponds with the number of URLs
        metadata = {**(metadata or {}), **kwargs}
        metadata_lengths = {k: len(v) for k, v in metadata.items()}
        if metadata_lengths:
            assert all(len(v) == len(urls) for v in metadata.values()), (
                f"Metadata must be of length {len(urls)}, "
                "but the number of elements in each field was:\n"
                f"{metadata_lengths}"
            )

        # Load all pages' HTML text into Documents
        doc_loader = AsyncHtmlLoader(urls)
        # Return list of Documents: await doc_loader.aload()
        # Return list of str: await doc_loader.afetch_all()
        docs = await doc_loader.aload()

        # Convert metadata to a list of dict with repeated keys (i.e., like JSON)
        if metadata:
            metadata = [
                {k: v[i] for k, v in metadata.items()} for i in range(len(urls))
            ]
            docs = await asyncio.gather(
                *[
                    await add_metadata_to_document(doc, metadata)
                    for doc, metadata in zip(docs, metadata)
                ]
            )

        return docs
    return (create_documents_from_urls,)


@app.cell
def __(
    Document,
    MarkdownTextSplitter,
    MarkdownifyTransformer,
    TokenCounter,
    add_metadata_to_document,
    asyncio,
):
    async def transform_and_chunk_docs(
        docs: list[Document],
        max_length: int = 8_192,
        avg_characters_per_token: int | float = 4.0,
        model_name: str | None = None,
    ) -> list[Document]:
        """
        Transform documents to Markdown and chunk based on document structure.

        Parameters
        ----------
        docs
            List of LangChain `Document`s to transform and chunk.
        max_length: int, default = 8,192
            Maximum number of tokens in a chunk. This should be set based
            on the embedding model's maximum input length.
        avg_characters_per_token: float, default = 4.0
            The approximated number of characters per token for
            the tokenizer for the given embedding model. If `model_name`
            is not given, this will be used to determine the maximum
            size of chunks (in number of characters).
        model_name: str, default = None
            Name of the embedding model that will be used for embedding
            the chunked documents (in a separate function). If provided,
            the model's tokenizer will be loaded to determine the model's
            maximum context length and use that to split the documents.
            The `max_length` will serve as an upper bound and the
            `avg_characters_per_token` will be ignored.

            For example: model_name="Alibaba-NLP/gte-base-en-v1.5"

        Returns
        -------
        List of Documents (which are the original documents, but
        chunked into smaller Documents). This list have equal to
        or greater number of elements than the input `docs`.
        """
        if model_name:
            token_counter = TokenCounter(model_name=model_name)
            token_counter.load_tokenizer()

        async def count_words(doc) -> int:
            return len(doc.page_content.split())

        async def count_characters(doc) -> int:
            return len(doc.page_content)

        async def count_tokens(doc, token_counter) -> int:
            return token_counter.count_tokens(doc.page_content)

        # Convert HTML to Markdown (and simplify it in the process)
        doc_transformer = MarkdownifyTransformer(
            # Remove Markdown formatting for images, links, strong, and emphasized text
            strip=["img", "a", "strong", "b", "em", "i"],
            # Default for bullets: "*+-", where each successive level
            # uses a different bullet character
            bullets="-*",
            # Set the assumed language for code blocks (i.e., "```python ...```")
            code_language="python",
            # Disable escaping Markdown formatting characters because this
            # document does not need to be rendered, only processed by the
            # embedding model.
            escape_asterisks=False,
            escape_underscores=False,
            escape_misc=False,
            # For more options, see:
            # https://github.com/matthewwithanm/python-markdownify/tree/develop
        )

        text_splitter = MarkdownTextSplitter(
            chunk_size=(
                max_length * avg_characters_per_token
                if not model_name
                else min(max_length, token_counter.max_length)
            ),
            length_function=(
                len if not model_name else token_counter.count_tokens
            ),
        )
        docs = await doc_transformer.atransform_documents(docs)
        chunked_docs = await text_splitter.atransform_documents(docs)
        chunked_docs = await asyncio.gather(
            *[
                add_metadata_to_document(
                    doc,
                    {
                        "word_count": await count_words(doc),
                        "token_count": await count_tokens(doc, token_counter),
                        "character_count": await count_characters(doc),
                    },
                )
                for doc in chunked_docs
            ]
        )
        return chunked_docs
    return (transform_and_chunk_docs,)


@app.cell
async def __(create_documents_from_urls, transform_and_chunk_docs):
    urls = [
        "https://churchofjesuschristtemples.org/payson-utah-temple/",
        "https://www.deeplearning.ai/the-batch/issue-270/",
        "https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/",
        "https://scikit-learn.org/stable/modules/clustering.html",
        "https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.explain.html",
        "https://docs.marimo.io/guides/plotting.html",
    ]

    raw_docs = await create_documents_from_urls(urls=urls)
    docs = await transform_and_chunk_docs(
        docs=raw_docs, model_name="Alibaba-NLP/gte-base-en-v1.5"
    )

    docs
    return docs, raw_docs, urls


@app.cell
def __(docs):
    # Determine the number of chunks per document
    num_chunks = {}
    for doc in docs:
        _ = num_chunks.setdefault(doc.metadata["source"], 0)
        num_chunks[doc.metadata["source"]] += 1
    num_chunks
    return doc, num_chunks


@app.cell
def __(mo):
    mo.md(
        """
        ## Embeddings and vector store

        Create embeddings and store in a vector database
        """
    )
    return


@app.cell
def __(HuggingFaceEmbeddings):
    embedding_models = [
        "Alibaba-NLP/gte-base-en-v1.5",
        "nomic-ai/nomic-embed-text-v1.5",
        "avsolatorio/GIST-small-Embedding-v0",
        "BAAI/bge-small-en-v1.5",
    ]
    model_name = embedding_models[0]
    model_kwargs = {
        "device": "cpu",
        # For models that require you to execute the configuration file
        # from their repositories on your local machine; ensure you trust
        # the repository (check its code on Hugging Face) before setting
        # trust_remote_code=True.
        "trust_remote_code": True,
    }
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        # For the the query_instruction and embed_instruction arguments,
        # use the HuggingFaceBgeEmbeddings class from langchain_community.embeddings.
        # query_instruction="",
        # embed_instruction="",
        show_progress=True,
    )
    return (
        embedding_models,
        embeddings,
        encode_kwargs,
        model_kwargs,
        model_name,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Similarity search

        Compare a user's query to the documents and return the most similar documents to the query.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Other experiments
        I chose not to use these methods, but I'm keeping them here for documentation.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Document retrieval: `WebBaseLoader`

        ```python
        from langchain_community.document_loaders import WebBaseLoader
        ```

        [`WebBaseLoader`](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html#) was not as accurate at extracting the meaningful parts of web pages compared to `AsyncHtmlLoader` (for page content) combined with `MarkdownifyTransformer`.

        In particular, `WebBaseLoader` returned poorly formatted documents, with lots of extra whitespace, especially around the header (navigation) and footer parts of web pages.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Embeddings models: `HuggingFaceBgeEmbeddings`

        ```python
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        # Note that the old import form (below) is deprecated,
        # I'm noting it here because you might still see it in documentation:
        # from langchain.embeddings import HuggingFaceBgeEmbeddings
        ```

        The Beijing Academy of Artificial Intelligence (BAAI) created BGE models: BAAI General Embeddings, which are medium-sized embedding models that provide strong performance.

        The `HuggingFaceBgeEmbeddings` class supports `query_instruction` and `embed_instruction` parameters to prefix each document, which is required for embedding models that were trained to differentiate between the query (search terms) and the documents to be searched when embedding both of them to compute similarity.

        The query prompt and embedding prompt are defined during training and are set by the model developers; they are not general-purpose prompts to be tuned.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Embeddings models: `langchain_huggingface` module

        ```python
        from langchain_huggingface import HuggingFaceEmbeddings
        ```

        I opted to use `langchain_community.embeddings.HuggingFaceBgeEmbeddings` instead because the dedicated `langchain_huggingface` package requires a ton of additional software whereas `langchain_community.embeddings.HuggingFaceBgeEmbeddings` requires only `sentence-transformers`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
