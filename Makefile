.ONESHELL:
.PHONY: requirements

PYTHON_INTERPRETER = python

run: requirements
	@echo "Starting the service..."
	$(PYTHON_INTERPRETER) -m api.run

data: requirements
	@echo "Downloading data..."
	$(PYTHON_INTERPRETER) -m src.data.download_raw_data
	@echo "Filtering data..."
	$(PYTHON_INTERPRETER) -m src.data.filter_raw_data
	@echo "Processing data..."
	$(PYTHON_INTERPRETER) -m src.data.build_text_dataset
	@echo "Splitting data..."
	$(PYTHON_INTERPRETER) -m src.data.make_train_validation_test_split
	@echo "Building Features and Data Model..."
	$(PYTHON_INTERPRETER) -m src.data.build_features_and_save_dataset
	@echo "Data built succesfully!"

requirements: .venv
	# source .venv/Scripts/activate
	# @echo "Installing required packages ..."
	# $(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	# $(PYTHON_INTERPRETER) -m pip install -r requirements.txt

.venv:
	@echo "Creating Virtual Environment..."
	$(PYTHON_INTERPRETER) -m venv .venv
	(	\
		source ./.venv/Scripts/activate; \
		$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel; \
		$(PYTHON_INTERPRETER) -m pip install -r requirements.txt; \
	)
	@echo "Virtual Environment Created..."
