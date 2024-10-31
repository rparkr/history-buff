import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        """
        # Website image screenshots with `playwright`

        Adapted from: [`playwright` GitHub README](https://github.com/microsoft/playwright-python)

        In this section, I implement a function to save 1280x720 screenshots of web pages, which can then be passed to a vision-language model to generate a description of the page that can be included in the text for the webpage's `Document` and then embedded with the rest of the page, enabling users to search for a page based on its appearance or feel (e.g., <span style="color: lightgreen; font-weight: bold">modern</span>, <span style="color: lightgreen; font-weight: bold">fluid</span>, or <span style="color: lightgreen; font-weight: bold">clean</span> vs. <span style="color: orange; font-weight: bold">dated</span>, <span style="color: orange; font-weight: bold">confusing</span>, or <span style="color: orange; font-weight: bold">cluttered</span>).
        """
    )
    return


@app.cell
async def __():
    import asyncio
    import datetime
    from pathlib import Path
    from urllib.parse import urlparse

    from playwright.async_api import async_playwright

    # Sample url list:
    urls = [
        "https://churchofjesuschristtemples.org/payson-utah-temple/",
        "https://www.deeplearning.ai/the-batch/issue-270/",
        "https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/",
        "https://scikit-learn.org/stable/modules/clustering.html",
        "https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.explain.html",
        "https://docs.marimo.io/guides/plotting.html",
    ]


    async def save_page_screenshots(
        urls: list[str], save_dir: str = "screenshots"
    ) -> list[str]:
        """Save screenshots of webpages."""
        # Create a directory to save the screenshots
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        async def save_screenshot_single_page(
            browser, url: str, save_dir: str
        ) -> str:
            """Save a screenshot of a single page and return the filepath"""
            # Get the page's domain to use for the filename
            page_domain = urlparse(url).netloc

            # Open a new page and go to the URL
            page = await browser.new_page()
            # Use a timeout of 10 seconds to avoid waiting too long for pages
            # that won't load. The default is 30 seconds.
            await page.goto(url, timeout=10_000)
            # At 1280 x 720 (the default resolution), the PNG format
            # results in file sizes about 2x smaller than of JPEG at quality=95,
            # so PNG is preferred. The runtime is nearly identical under
            # either method.
            filepath = save_dir / f"{page_domain}_{datetime.date.today()}.png"
            await page.screenshot(path=filepath)
            return filepath

        async with async_playwright() as p:
            browser_type = p.chromium  # other options: p.firefox, p.webkit
            browser = await browser_type.launch(headless=True)
            # Use asyncio.gather() to concurrently run an operation on a list
            screenshot_filepaths = await asyncio.gather(
                *[
                    save_screenshot_single_page(browser, url, save_dir)
                    for url in urls
                ]
            )
            # Free resources by closing the browser when finished
            await browser.close()

        return screenshot_filepaths


    await save_page_screenshots(urls=urls)
    return (
        Path,
        async_playwright,
        asyncio,
        datetime,
        save_page_screenshots,
        urlparse,
        urls,
    )


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
