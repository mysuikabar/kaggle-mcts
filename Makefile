USER_NAME = suikabar
DATASET_TITLE = mcts-models

.PHONY: create_dataset push_dataset

create_dataset:
	@if [ -z "$(dataset_dir)" ]; then \
		echo "Error: dataset_dir must be specified."; \
		echo "Usage: make create_dataset dataset_dir=<path_to_dataset>"; \
		exit 1; \
	fi
	python src/tools/push_dataset.py \
		--dataset_dir $(dataset_dir) \
		--user_name $(USER_NAME) \
		--title $(DATASET_TITLE) \
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
		--title $(DATASET_TITLE)