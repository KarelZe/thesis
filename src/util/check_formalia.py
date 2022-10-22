import glob
import os
import re

import typer


def check_formulae(file_name: str, file_contents: str) -> None:
    """
    Check if formula contains \times or \quad.

    Args:
        file_name (str): file name
        file_contents (str): contents of file
    """
    matches = re.findall(r"\\times|\\quad", file_contents)
    if matches:
        msg = typer.style(
            f"{file_name}: {matches} (prefer \\cdot over \\times)",
            fg=typer.colors.YELLOW,
        )
        typer.echo(msg)


def check_acronyms(file_name: str, file_contents: str, acronyms: list) -> None:
    """
    Check if there are acronyms in text that don't use \gls{acr} or \gls{acr} wrapping.

    Args:
        file_name (str): file name
        file_contents (str): content of file
        acronyms (list): list with acronyms
    """
    matches = []
    for acronym in acronyms:
        match = re.findall(f"{acronym} |{acronym}s ", file_contents.lower())
        if match:
            matches.extend(match)

    if matches:
        msg = typer.style(
            f"{file_name}: {matches} (use \\gls{{...}} for acroynm)",
            fg=typer.colors.YELLOW,
        )
        typer.echo(msg)


def check_hyphens(file_name: str, file_contents: str, vocabulary: list) -> None:
    """
    Check if there are versions of the same word with and w/o hyphen.

    E. g., semi-supervised and semisupervised. (true) semi supervised (false).

    Args:
        file_name (str): file name
        file_contents (str): file content
        vocabulary (list): vocabulary in document
    """
    hyphenated_words = re.findall(r"\w+(?:-\w+)+", file_contents)
    hyphenated_words = [x.lower() for x in hyphenated_words]

    hyphenated_words_wo_hyphen = [re.sub(r"-", "", word) for word in hyphenated_words]

    matches = []
    for word in hyphenated_words_wo_hyphen:
        if word in vocabulary:
            matches.append(word)

    if matches:
        msg = typer.style(
            f"{file_name}: {matches} (word w and w/o hyphen)", fg=typer.colors.YELLOW
        )
        typer.echo(msg)


def check_refs(file_name: str, file_contents: str) -> None:
    """
    Check if there are references to the tables, figures, appendix in lower-case letters.

    Args:
        file_name (str): file name
        file_contents (str): file contents
    """
    r = re.compile(
        r"\sformula\s|\sequation\s|\sseq.\s|\sfigure\s|\sfig.\s|\stable\s|\stab.\s|\sappendix\s|\sapp.\s"
    )
    matches = re.findall(r, file_contents)
    if matches:
        msg = typer.style(
            f"{file_name}: {matches} (prefer capitalized e. g., Eq.)",
            fg=typer.colors.YELLOW,
        )
        typer.echo(msg)


def loc_files() -> dict:
    """
    Locate all urls in .tex files located in /reports dir.

    Returns:
        dict: Dict with filename as key and file contents as values.
    """

    os.chdir("../../reports")
    files = glob.glob("./**/*.tex", recursive=True)

    typer.echo(f"files checked: {files}")

    # merge files into dict
    matches_per_file = {}

    for fname in files:
        with open(fname, "rt", encoding="utf8") as infile:
            file_contents = infile.read()
            matches_per_file[fname] = file_contents

    print(matches_per_file.keys())
    return matches_per_file


def get_acronyms(files: dict) -> list:
    """
    Get acroynms from .tex file in lower-case.

    Args:
        files (dict): dict with filenames and file contents

    Returns:
        list: list with acroynms
    """
    rough_matches = re.findall(r"\\newacronym.+", files.get(".\\expose.tex"))
    refined_matches = re.findall(r"\{\b[^{}]*?}", "".join(rough_matches))
    # remove brackets and filter every third to skip over long form, convert to lower-case
    acronyms = [re.sub(r"[\{\}]", "", part.lower()) for part in refined_matches[::3]]
    return acronyms


def get_vocabulary(files: dict) -> list:
    """
    Gets vocabulary in lower-case from files.

    Args:
        files (dict): dict with filename and file contents

    Returns:
        list: list in lower-case of vocabulary
    """
    vocab = []
    for _, file_contents in files.items():
        words_in_file = re.findall(r"\b\S+\b", file_contents)
        vocab.extend(words_in_file)
    vocab = [x.lower() for x in vocab]
    return vocab


def check_formalia(files: dict, vocabulary: list, acronyms: list):
    """
    Check if formalia is fullfilled.

    Args:
        files (dict): dict with filename as key and list of urls
    """
    for file_name, file_contents in files.items():
        check_formulae(file_name, file_contents)
        check_refs(file_name, file_contents)
        check_acronyms(file_name, file_contents, acronyms)
        check_hyphens(file_name, file_contents, vocabulary)

    msg = typer.style("All files checked. 🍰✨🚀", fg=typer.colors.GREEN)
    typer.echo(msg)


def main():
    """
    Locate and check files.
    """
    files = loc_files()
    acroynms = get_acronyms(files)
    vocabulary = get_vocabulary(files)
    check_formalia(files, vocabulary, acroynms)


if __name__ == "__main__":
    typer.run(main)
