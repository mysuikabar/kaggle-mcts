import json
import subprocess
import tempfile
from pathlib import Path

import click
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi

REPO_ROOT = Path(__file__).resolve().parents[1]


def combine_files(file_path: Path, output_dir: Path) -> Path:
    """
    Combine Python files using stickytape.
    """
    output_file = output_dir / "combined.py"
    subprocess.run(
        [
            "stickytape",
            str(file_path),
            "--add-python-path",
            str(REPO_ROOT / "src"),
            "--output-file",
            str(output_file),
        ],
        check=True,
    )
    return output_file


def convert_py_to_ipynb(file_path: Path, output_dir: Path) -> Path:
    """
    Convert a Python file to Jupyter notebook format.
    """
    python_code = file_path.read_text()
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [nbformat.v4.new_code_cell(python_code)]

    output_file = output_dir / "notebook.ipynb"
    with open(output_file, "w") as f:
        nbformat.write(nb, f)

    return output_file


def create_metadata(
    kernel_dir: Path,
    user_name: str,
    title: str,
    notebook_file: Path,
    competition: str,
    dataset: str | None = None,
) -> None:
    """
    Create metadata for the Kaggle kernel.
    """
    metadata = {
        "id": f"{user_name}/{title}",
        "title": title,
        "code_file": notebook_file.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [competition],
    }

    if dataset:
        metadata["dataset_sources"] = [dataset]

    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f)


def upload_kernel(kernel_dir: Path) -> None:
    """
    Upload the kernel to Kaggle.
    """
    api = KaggleApi()
    api.authenticate()
    api.kernels_push(kernel_dir)


@click.command()
@click.option("--file_path", type=Path, required=True)
@click.option("--user_name", type=str, required=True)
@click.option("--title", type=str, required=True)
@click.option("--competition", type=str, required=True)
@click.option("--dataset", type=str)
def main(
    file_path: Path, user_name: str, title: str, competition: str, dataset: str | None
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        try:
            combined_py = combine_files(file_path, tmp_dir_path)
            notebook_file = convert_py_to_ipynb(combined_py, tmp_dir_path)
            create_metadata(
                kernel_dir=tmp_dir_path,
                user_name=user_name,
                title=title,
                notebook_file=notebook_file,
                competition=competition,
                dataset=dataset,
            )
            upload_kernel(tmp_dir_path)
            click.echo(
                f"Successfully pushed to Kaggle: https://www.kaggle.com/code/{user_name}/{title}"
            )
        except Exception as e:
            click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
