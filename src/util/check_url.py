import glob
import os
import re
from urllib.request import urlopen
from urllib.error import HTTPError

import typer


def loc_urls() -> dict:
    """
    Locate all urls in .bib and .tex files located in /reports dir.

    Returns:
        dict: Dict with filename as key and list of urls as values.
    """

    # adapted from https://stackoverflow.com/a/2102648/5755604
    r = re.compile(
        r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-,}/]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
    )

    # find all .bib and .tex files
    os.chdir("../../reports")
    files = glob.glob("./**/*.tex", recursive=True) + glob.glob(
        "./**/*.bib", recursive=True
    )

    typer.echo(f"files checked: {files}")

    # merge files into dict
    matches_per_file = {}

    for fname in files:
        with open(fname, "rt", encoding="utf8") as infile:
            file_contents = infile.read()
            matches = re.findall(r, file_contents)

            # write only matches with urls
            if matches:
                matches_per_file[fname] = matches

    return matches_per_file


def check_urls(urls: dict):
    """
    Check if urls can be resolved.

    Args:
        urls (dict): dict with filename as key and list of urls
    """
    for filename, urls in urls.items():
        for url in urls:
            try:
                urlopen(url)
            except HTTPError as err:
                msg = typer.style(f"{filename}: {url} ({err})", fg=typer.colors.YELLOW)
                typer.echo(msg)

    msg = typer.style("All URLs checked. 🍰✨🚀", fg=typer.colors.GREEN)
    typer.echo(msg)


def main():
    """
    Locate and check urls.
    """
    urls = loc_urls()
    check_urls(urls)


if __name__ == "__main__":
    typer.run(main)