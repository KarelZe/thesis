# -*- coding: utf-8 -*-
"""
Script to pre-process the raw data set.

See `notebooks/` for further details.
"""
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: click.Path, output_filepath: click.Path) -> None:
    """
    Run data processing scripts to turn raw data from (../raw) into\
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        input_filepath (click.Path): input file
        output_filepath (click.Path): output file
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
