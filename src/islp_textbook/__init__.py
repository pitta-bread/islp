import argparse
import importlib
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chapter",
        type=int,
        required=True,
        help="Chapter number of the lab to run",
    )
    args = parser.parse_args(argv)

    module_name = f"islp_textbook.lab_{args.chapter}"
    try:
        lab = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            parser.error(f"no lab exists for chapter {args.chapter}")
        raise

    lab.main()
    return 0
