#!/usr/bin/env python
# coding: utf-8


# %% markdown
# # Image captioning
# This section shows how to caption an image using various
# small vision models through Ollama.
#
# **Purpose:** replace image URLs with a detailed description of
# each image prior to embedding the image, and also to add an overview
# to the beginning of the webpage's text contents explaining the overall
# layout, design, and feel of the website, so that text can also
# be included in the embedding. The playwright library can
# take screenshots of pages, which could then be passed to
# the vision-language model for captioning.
#
# An alternative would be to generate the embeddings using
# a CLIP-style model that jointly embeds text and images.
#
# ## Takeaways:
# - [moondream2](https://huggingface.co/vikhyatk/moondream2) is a great model, the fastest among those compared, but its quality is better through [Hugging Face](https://huggingface.co/spaces/vikhyatk/moondream2)
# - [llava-phi3](https://ollama.com/library/llava-phi3) is about 2x slower than moondream, but provides slightly better results (from Ollama, not Hugging Face)
# - [minicpm-v](https://ollama.com/library/minicpm-v) is much slower (about 6x slower than moondream) but it provided excellent results
# %%
import base64

import httpx
import ollama

image_bytes = httpx.get(
    "https://raw.githubusercontent.com/rparkr/baby-names/refs/heads/main/images/baby_names_streamlit.png"
).content

# CLI prompt: This is a screenshot of a web page. Describe the page, its contents, layout, design, and feel. /teamspace/studios/this_studio/data/baby_names_streamlit.png

stream = ollama.generate(
    model="moondream",  # "moondream" (1.8B params, 1.7 GB), "llava-phi3" (3.8B params, 2.9 GB), "minicpm-v" (8B params, 5.5 GB)
    prompt="This is a screenshot of a web page. Describe the page, its contents, layout, design, and feel.",
    images=[base64.b64encode(image_bytes)],
    stream=True,
)

# 37s for moondream, 30s to first token
# 101s for llava-phi3, 55s until first token
# 289s for minicpm-v, 204s until first token  --> best response


for chunk in stream:
    print(chunk["response"], end="", flush=True)


# %% markdown
# # Code explanation
# This section compares the performance of various
# small, code-focused LLMs at summarizing blocks of code.
#
# **Purpose:** replace code blocks on a webpage with a description
# of the purpose of that code block. The resulting text can
# then be used in place of (or alongside) the code block
# prior to creating embeddings for the page.
#
# This function could also be used for a text-to-speech
# application, since it can be difficult to understand
# generated speech that simply reads the tokens and whitespace
# of a code block rather than explaining what that code block does.

# %%
import json
import re
import time

import httpx
from markdownify import markdownify as md
import ollama


pages = [
    "https://docs.marimo.io/guides/working_with_data/plotting.html",
    "https://fastapi.tiangolo.com/tutorial/body/#declare-it-as-a-parameter",
]


def url_to_md(url: str = "") -> str:
    """
    Convert a webpage to markdown format using markdownify.
    """
    page = httpx.get(url).text
    md_page = md(page, code_language="python", heading_style="ATX")
    return md_page


def extract_code(markdown_page: str = "") -> list[str]:
    """
    Extract code blocks from a Markdown-formatted page.
    """
    matches = re.findall(
        pattern=r"```python.*?```", string=markdown_page, flags=re.DOTALL
    )
    return matches


def explain_code(
    code_strings: list[str],
    context: list[int] | None = None,
    model: str = "qwen2.5-coder:1.5b",
    prompt: str = "Explain the the following block of Python code, along with its purpose and what it achieves: ",
) -> list[str]:
    """Explain blocks of code using an LLM"""
    if not isinstance(code_strings, list):
        code_strings = [code_strings]
    code_strings_count = len(code_strings)
    explanations = []
    for n, code_string in enumerate(code_strings):
        start_time = time.perf_counter()
        explanations.append(
            ollama.generate(
                model=model,
                prompt=prompt + code_string,
                context=context,
            )
        )
        duration = time.perf_counter() - start_time
        print(
            f"Model: {model}, code block: {n+1}/{code_strings_count}, time: {duration // 60:.0f}m {duration % 60:.2f}s"
        )
    return explanations

model_explanations = {}
for model in [
    "qwen2.5-coder:1.5b",
    "yi-coder:1.5b",
    "codegemma:2b",
]:
    model_explanations[model] = explain_code(
        model=model,
        code_strings=extract_code(url_to_md(pages[-1]))
    )
    
with open("model_explanations.json", mode="wt", encoding="utf8") as file:
    json.dump(model_explanations, file)
