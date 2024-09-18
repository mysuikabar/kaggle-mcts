USER_NAME = suikabar
COMPETITION = um-game-playing-strength-of-mcts-variants
DATASET = mcts-models

.PHONY: create_dataset push_dataset push_notebook

create_dataset:
	@if [ -z "$(dataset_dir)" ]; then \
		echo "Error: dataset_dir must be specified."; \
		echo "Usage: make create_dataset dataset_dir=<path_to_dataset>"; \
		exit 1; \
	fi
	python src/tools/push_dataset.py \
		--dataset_dir $(dataset_dir) \
		--user_name $(USER_NAME) \
		--title $(DATASET) \
		--new

push_dataset:
	@if [ -z "$(dataset_dir)" ]; then \
		echo "Error: dataset_dir must be provided."; \
		echo "Usage: make push_dataset dataset_dir=<path_to_dataset>"; \
		exit 1; \
	fi
	python src/tools/push_dataset.py \
		--dataset_dir $(dataset_dir) \
		--user_name $(USER_NAME) \
		--title $(DATASET)

push_notebook:
	@if [ -z "$(file_path)" ]; then \
		echo "Error: notebook_path must be provided."; \
		echo "Usage: make push_kernel notebook_path=<path_to_notebook> title=<kernel_title> competition=<competition_name> [dataset=<dataset_name>]"; \
		exit 1; \
	fi
	@if [ -z "$(title)" ]; then \
		echo "Error: title must be provided."; \
		echo "Usage: make push_kernel notebook_path=<path_to_notebook> title=<kernel_title> competition=<competition_name> [dataset=<dataset_name>]"; \
		exit 1; \
	fi
	python src/tools/push_notebook.py \
		--file_path $(file_path) \
		--user_name $(USER_NAME) \
		--title $(title) \
		--competition $(COMPETITION) \
		--dataset $(USER_NAME)/$(DATASET)
