import json
from pathlib import Path

import click
from kaggle.api.kaggle_api_extended import KaggleApi


def create_metadata(
    dataset_dir: Path, user_name: str, title: str, description: str | None = None
) -> None:
    metadata = {
        "title": title,
        "id": f"{user_name}/{title}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    if description:
        metadata["description"] = description

    with open(dataset_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f)


def upload_dataset(dataset_dir: Path, new: bool) -> None:
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=dataset_dir, public=False, convert_to_csv=False, dir_mode="zip"
        )
    else:
        api.dataset_create_version(
            folder=dataset_dir, version_notes="", convert_to_csv=False, dir_mode="zip"
        )


@click.command()
@click.option("--dataset_dir", type=Path, required=True)
@click.option("--user_name", type=str, required=True)
@click.option("--title", type=str, required=True)
@click.option("--description", type=str)
@click.option("--new", is_flag=True, default=False)
def main(
    dataset_dir: Path, user_name: str, title: str, description: str | None, new: bool
) -> None:
    create_metadata(dataset_dir, user_name, title, description)
    upload_dataset(dataset_dir, new)
    click.echo("Dataset upload completed.")


if __name__ == "__main__":
    main()
