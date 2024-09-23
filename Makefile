include .env

.PHONY: create_dataset push_dataset push_notebook

create_dataset:
	@if [ -z "$(1)" ] || [ -z "$(2)" ]; then \
		echo "Error: dataset_dir and title must be specified."; \
		echo "Usage: make create_dataset <dataset_dir> <dataset_title>"; \
		exit 1; \
	fi
	python src/tools/push_dataset.py \
		--dataset_dir $(1) \
		--user_name $(USER_NAME) \
		--title $(2) \
		--new

push_dataset:
	@if [ -z "$(1)" ] || [ -z "$(2)" ]; then \
		echo "Error: dataset_dir and title must be provided."; \
		echo "Usage: make push_dataset <dataset_dir> <dataset_title>"; \
		exit 1; \
	fi
	python src/tools/push_dataset.py \
		--dataset_dir $(1) \
		--user_name $(USER_NAME) \
		--title $(2)

push_notebook:
	@if [ -z "$(1)" ] || [ -z "$(2)" ]; then \
		echo "Error: file_path and title must be provided."; \
		echo "Usage: make push_notebook <file_path> <kernel_title> [dataset]"; \
		exit 1; \
	fi
	python src/tools/push_notebook.py \
		--file_path $(1) \
		--user_name $(USER_NAME) \
		--title $(2) \
		--competition $(COMPETITION) \
		$(if $(3),--dataset $(USER_NAME)/$(3))
