import sys

# Argument parsing
from argparse import ArgumentParser, Namespace
from pathlib import Path

from dotenv import load_dotenv

import pandas as pd

# Configuration
from iconfig.iconfig import iConfig

from loguru import logger

from multiprocessing import Process

from md.ui.display import start
from md.model.version import Versions
from md.preprocess.preprocess import all_prepared, preprocess

def get_args() -> Namespace:
    """Reads command line arguments and returns a Namespace object with them

    Returns:
        Namespace: Namespace object with the command line arguments
    """
    parser = ArgumentParser(
        prog='mapdisplay.py',
        description='Entry point for the Health Monitor that allows training, inference, retraining, and EDA'
    )

    # Add a positional argument for the task
    parser.add_argument(
        "positional_version",
        nargs="?",  # Optional positional argument
        default=None,
        help="Version of the map"
    )

    # Add the -v/--version option
    parser.add_argument(
        "-v", "--version",
        action="store",
        dest="version",
        help="Version of the map"

    )

    # Add the -f/--force option
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        dest="force",
        help="Force preprocessing even if data is already prepared"
    )

    # Add the -rc/--recreate-container option
    parser.add_argument(
        "-rc", "--recreate-container",
        action="store_true",
        dest="recreate_container",
        help="Recreate container before preprocessing"
    )

    args = parser.parse_args()

    # Use the positional argument if -t/--task is not provided
    if args.version is None:
        args.version = args.positional_version

    # Ensure a task is provided
    if args.version is None:
        parser.error("No version provided. Use '<version>' or '-v <version>' to specify a version.")

    return args

def safe_exit(config: iConfig, process: Process, timeout: int):
    """Wait for background preprocessing to finish or timeout.
    
    Args:
        process: The multiprocessing.Process object (or None if not running in background)
        timeout: Maximum seconds to wait before force-terminating
    """
    if process is None or not config("preprocess.background", default=False):
        logger.debug("No background preprocessing to wait for")
        return
    
    start_time = pd.Timestamp.now()
    t_status_time = start_time
    logger.info("Waiting for preprocessing to finish...")
    
    while process.is_alive():
        elapsed = (pd.Timestamp.now() - t_status_time).total_seconds()
        if elapsed > 60:
            logger.info("Still waiting for preprocessing to finish...")
            t_status_time = pd.Timestamp.now()
        
        elapsed_total = (pd.Timestamp.now() - start_time).total_seconds()
        if elapsed_total > timeout:
            logger.error(f"Preprocessing timeout exceeded ({timeout}s). Terminating.")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                logger.error("Process did not terminate gracefully. Killing.")
                process.kill()
                process.join(timeout=5)
            return
        
        process.join(timeout=1)  # Check every 1 second

# Main
if __name__ == '__main__':

    # Load environment variables
    load_dotenv('.env')
    config = iConfig()

    # Get the command line arguments
    args = get_args()

    # Find version information
    if args.version:
        version = Versions().get_version(id=args.version)
    else:
        version = Versions().get_newest_version()
    
    if version is None:
        logger.error("No version information found. Please specify a version using the -v/--version option or ensure a .version file exists.")
        sys.exit(1)
    
    recreate_container = args.recreate_container
    
    if not all_prepared(version) or args.force:
        logger.info(f"Preprocessing data for version: {version.id} (Created at: {version.created_at}, Mapfile: {version.mapfile})")
        process = preprocess(version, recreate_container=recreate_container, background=config("preprocess.background", default=False))
    else:
        process = None

    logger.info(f'Mapdisplay starting up with version: {version.id} (Created at: {version.created_at}, Mapfile: {version.mapfile})')

    start()

    safe_exit(config=config, process=process, timeout=config("preprocess.timeout", default=300))
