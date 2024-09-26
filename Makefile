include .env

.PHONY: create_dataset push_dataset push_notebook

create_dataset:
	@if [ -z "$(dataset_dir)" ] || [ -z "$(title)" ]; then \
		echo "Error: dataset_dir and title must be specified."; \
		echo "Usage: make create_dataset dataset_dir=<dataset_dir> title=<title>"; \
		exit 1; \
	fi
	python tools/push_dataset.py \
		--dataset_dir $(dataset_dir) \
		--user_name $(USER_NAME) \
		--title $(title) \
		--new

push_dataset:
	@if [ -z "$(dataset_dir)" ] || [ -z "$(title)" ]; then \
		echo "Error: dataset_dir and title must be provided."; \
		echo "Usage: make push_dataset dataset_dir=<dataset_dir> title=<title>"; \
		exit 1; \
	fi
	python tools/push_dataset.py \
		--dataset_dir $(dataset_dir) \
		--user_name $(USER_NAME) \
		--title $(title)

push_notebook:
	@if [ -z "$(file_path)" ] || [ -z "$(title)" ]; then \
		echo "Error: file_path and title must be provided."; \
		echo "Usage: make push_notebook file_path=<file_path> title=<title> [dataset=<dataset>]"; \
		exit 1; \
	fi
	python tools/push_notebook.py \
		--file_path $(file_path) \
		--user_name $(USER_NAME) \
		--title $(title) \
		--competition $(COMPETITION) \
		$(if $(dataset),--dataset $(USER_NAME)/$(dataset))