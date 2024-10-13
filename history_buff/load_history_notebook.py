import marimo

__generated_with = "0.9.8"
app = marimo.App(width="medium", app_title="Load history", css_file="")


@app.cell
def __():
    import glob
    import logging
    from pathlib import Path
    import platform  # determine operating system. See: https://note.nkmk.me/en/python-platform-system-release-version/
    import sqlite3
    from typing import Literal

    import marimo as mo
    import polars as pl

    logger = logging.getLogger(__name__)
    return Literal, Path, glob, logger, logging, mo, pl, platform, sqlite3


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Load history
        - For Google Chrome, history is located at: `/home/user/.config/google-chrome/Default/History`, and the relevant tables are `urls` and `visits`
        - For Firefox, history is located at: `/home/user/.mozilla/firefox/*.default-release/history.sqlite` (where `*.default-release` is replaced with the default user's profile folder) and the relevant tables are `moz_places` and `moz_historyvisits`

        **Reference:** [NXLog article](https://docs.nxlog.co/integrate/browser-history.html)
        """
    )
    return


@app.cell
def __(Literal, Path, glob, logger, pl, platform, sqlite3):
    class BrowserHistoryDB:
        def __init__(
            self,
            browser: Literal["chrome", "firefox"] = "chrome",
            db_filepath: str | None = None,
        ):
            """
            Manage connections to the web browsing history database.

            Parameters
            ----------
            browser : str, one of: {"chrome", "firefox"}, default: "chrome"
                The browser you want to load the browsing history from.

            db_filepath : str, None. default: None
                Filepath to the SQLite database with web browsing history. If `None`,
                then the filepath will be determined using this class's method:
                `get_path_to_history_database()`.
            """
            if browser not in ("chrome", "firefox"):
                raise ValueError(
                    "The only supported browsers are 'chrome' and 'firefox', "
                    f"but '{browser}' was given."
                )
            self.browser = browser
            if not db_filepath:
                self.db_filepath = self.get_path_to_history_database()
            else:
                self.db_filepath = db_filepath

            # Create column name mapping for the browsers
            self.db_columns = {
                "chrome": {
                    "id": "id",
                    "url": "url",
                    "title": "title",
                    "visit_count": "visit_count",
                    "last_visit_time": "last_visit_time",
                    "visit_time": "visit_time",
                    "visit_duration": "visit_duration",
                },
                "firefox": {
                    "id": "id",
                    "url": "url",
                    "title": "title",
                    "visit_count": "visit_count",
                    "last_visit_time": "last_visit_date",
                    "visit_time": "visit_date",
                    # Firefox's database does not seem to track visit duration
                },
            }

        def execute_query(
            self,
            query,
            parameters: dict | None = None,
            columns_only: bool = False,
            read_only: bool = True,
        ) -> tuple[list, list]:
            """
            Execute a query and return its result as a tuple of (rows, columns)

            Parameters
            ----------
            query : str
                SQL query to execute
            parameters : dict, default: None
                Parameters to pass to the query as named parameters, where the
                query contains placeholder values in the form of ":placeholder"
                and the parameters dict has a key for each placeholder value,
                for example: {"placeholder": "value
                See: https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders
            columns_only : bool, default: False
                If True, return only the column names
            read_only: bool, default: True
                If True, the connection is opened in read-only mode.

            Returns
            -------
            tuple: (rows, columns), if columns_only is False, otherwise a list of column names
            """
            db_filepath = self.db_filepath
            # Open in read-only mode
            # See: https://docs.python.org/3/library/sqlite3.html#how-to-work-with-sqlite-uris
            if read_only:
                db_filepath = "file:" + str(db_filepath) + "?mode=ro"
            with sqlite3.connect(db_filepath, uri=read_only) as conn:
                # SQLite allows executing from the connection rather than the cursor
                if parameters:
                    result = conn.execute(query, parameters)
                else:
                    result = conn.execute(query)
                columns = [row[0] for row in result.description]
                if columns_only:
                    return columns
                # Rows will be included, so fetch their values as well
                rows = result.fetchall()
            return rows, columns

        def get_column_names(self, table_name):
            """Return the column names from a table in a SQLite database."""
            # In SQLite you cannot use placeholders with table or column names
            # so I use string interpolation
            rows, columns = self.execute_query(f"PRAGMA table_info({table_name})")
            # The PRAGMA function returns a table where the second column has the
            # column name (columns: ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'])
            # which mean: column id, name, type, not null, default value, primary key
            return [row[1] for row in rows]

        def get_path_to_history_database(self) -> Path:
            """Return the path to the web browsing history database."""
            system_os = platform.system().lower()
            user_home = Path().home()
            history_location = {
                "linux": {
                    "chrome": user_home / ".config/google-chrome/Default/History",
                    "firefox": f"{user_home}/.mozilla/firefox/*.default-release",
                },
                # "Darwin" is macOS
                "darwin": {
                    "chrome": user_home
                    / "Library/Application Support/Google/Chrome/Default/History",
                    "firefox": f"{user_home}/Library/Application Support/Firefox/Profiles/*.default-release",
                },
                "windows": {
                    "chrome": Path(
                        rf"{user_home}\AppData\Local\Google\Chrome\User Data\Default\History"
                    ),
                    "firefox": rf"{user_home}\AppData\Roaming\Mozilla\Firefox\Profiles\*.default-release",
                },
            }
            # determine the default profile, assumed to be the first one
            if self.browser == "firefox":
                return (
                    Path(glob.glob(history_location[system_os][self.browser])[0])
                    / "places.sqlite"
                )
            return history_location[system_os][self.browser]

        def load_history_database(self) -> pl.DataFrame:
            """
            Load the SQLite database with web browsing history and return as a DataFrame.

            Returns
            -------
            pl.DataFrame: dataframe with the database contents.
            """
            tables = {
                "chrome": {"url_table": "urls", "visit_table": "visits"},
                "firefox": {
                    "url_table": "moz_places",
                    "visit_table": "moz_historyvisits",
                },
            }
            url_table = tables[self.browser][
                "url_table"
            ]  # URL-level table (all distinct)
            visit_table = tables[self.browser][
                "visit_table"
            ]  # visit-level table (duplicate URLs allowed)

            # Google Chrome's timestamps are in units of microseconds
            # since 1-Jan-1601, while Firefox's are microsecond
            # since 1-Jan-1970 (the Unix epoch)
            # See: https://stackoverflow.com/questions/20458406/what-is-the-format-of-chromes-timestamps
            chrome_adjustment = (
                "strftime('%s', '1601-01-01')" if self.browser == "chrome" else 0
            )

            # In SQLite you cannot use placeholders for table or column names
            # so I use string interpolation
            query = f"""
            select
                {self.db_columns[self.browser]["id"]} as id,
                {self.db_columns[self.browser]["url"]} as url,
                {self.db_columns[self.browser]["title"]} as title,
                {self.db_columns[self.browser]["visit_count"]} as visit_count,
                strftime('%Y-%m-%d %H:%M:%f', ({self.db_columns[self.browser]["last_visit_time"]} / 1e6) + {chrome_adjustment}, 'unixepoch') as last_visit_time
            from
                {url_table}
            """
            logger.info("SQL query:\n\n%s", query)
            with sqlite3.connect(self.db_filepath) as con:
                return pl.read_database(query=query, connection=con).with_columns(
                    last_visit_time=pl.col("last_visit_time").str.to_datetime()
                )
            # Explicitly use row orientation since that is how SQLite
            # returns the data with cursor.fetchall().
            # rows, columns = self.execute_query(query)
            # return pl.DataFrame(
            #     data=rows, schema=columns, orient="row"
            # ).with_columns(
            #     last_visit_time=pl.col("last_visit_time").str.to_datetime()
            # )
    return (BrowserHistoryDB,)


@app.cell
def __(mo):
    browser_selection = mo.ui.dropdown(
        options=["chrome", "firefox"], value=None, allow_select_none=True
    )
    table_selection = mo.ui.dropdown(
        options=["urls", "visits", "moz_places", "moz_historyvisits"],
        value=None,
        allow_select_none=True,
    )
    return browser_selection, table_selection


@app.cell
def __(browser_selection, mo, table_selection):
    _md = mo.md(
        f"""
        ### Preview browsing history tables

        Browser: {browser_selection}  
        Table: {table_selection}
        """
    )
    _md
    return


@app.cell
def __(Path, browser_selection, mo, pl, sqlite3, table_selection):
    mo.stop(
        predicate=not (browser_selection.value and table_selection.value),
        output="Select the browser and table",
    )

    if browser_selection.value == "chrome":
        db_filepath = Path.home() / ".config/google-chrome/Default/History"
    else:
        db_filepath = (
            Path.home()
            / "/home/ryan/.mozilla/firefox/0f8ikj9p.default-release/places.sqlite"
        )
    # query = f"PRAGMA table_info({table_name})"  # describe the table's columns
    query = f"select * from {table_selection.value}"  # sample the table
    with sqlite3.connect(db_filepath) as con:
        # result = con.execute(query)
        _df = pl.read_database(query=query, connection=con)

    with mo.capture_stdout() as buffer:
        with pl.Config(tbl_rows=20, tbl_width_chars=1000, tbl_cols=20):
            print(_df)
    _output = mo.plain_text(buffer.getvalue())
    _output
    return buffer, con, db_filepath, query


@app.cell
def __(BrowserHistoryDB):
    history_db = BrowserHistoryDB(browser="chrome")
    df = history_db.load_history_database()
    df
    return df, history_db


@app.cell
def __():
    # Columns to use:
    # id, url, title, visit_count, visit_date (time), visit_duration, last_visit_date (time)
    return


@app.cell
def __(BrowserHistoryDB):
    history_db_firefox = BrowserHistoryDB(browser="firefox")
    df_firefox = history_db_firefox.load_history_database()
    df_firefox
    return df_firefox, history_db_firefox


if __name__ == "__main__":
    app.run()
