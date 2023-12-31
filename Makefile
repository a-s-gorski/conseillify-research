.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = conseillify-research
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m nltk.downloader all

## Setup file structure for storing data
folder_structure:
	mkdir -p {data/dataset,data/external,data/predictions,data/processed/,data/predictions/candidate_generation,data/predictions/learn_to_rank,data/raw,data/predictions/diversification}
## Make Dataset
# data: requirements
data:
	# export $(xargs < .env)
	# kaggle login
	# kaggle datasets download adamsebastiangorski/spotifysongfeatures -p data/external
	# unzip data/external/spotifysongfeatures.zip -d data/raw/spotifysongfeatures
	# aicrowd login
	# aicrowd dataset download --challenge spotify-million-playlist-dataset-challenge -o data/external
	# unzip data/external/spotify_million_playlist_dataset.zip -d data/raw/spotify_million_playlist_dataset
	# unzip data/external/spotify_million_playlist_dataset_challenge.zip -d data/raw/spotify_million_playlist_dataset_challange
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

features:
	$(PYTHON_INTERPRETER) -m src.features.build_features data/processed data/dataset

generate_candidates:
	$(PYTHON_INTERPRETER) -m src.models.candidate_generation.generate_candidates data/dataset data/processed data/predictions/candidate_generation models/candidate_generation

generate_candidates_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.candidate_generation.generate_candidates_pipeline \
	models/candidate_generation/candidate_generator.pkl \
	data/dataset/dataset_lightfm \
	data/processed/songs_encodings.csv \
	data/processed/songs_features.csv \
	data/processed/playlists.csv

test_generate_candidates:
	$(PYTHON_INTERPRETER) -m tests.models.candidate_generation.test_candidate_generation \
	models/candidate_generation/candidate_generator.pkl \
	data/dataset/dataset_lightfm \
	data/processed/songs_encodings.csv \
	data/processed/songs_features.csv \
	data/processed/playlists.csv \
	data/processed/test_playlists.csv \
	reports/eval_generate_candidates

rank:
	$(PYTHON_INTERPRETER) -m src.models.learn_to_rank.make_rank data/predictions/candidate_generation \
	models/learn_to_rank data/predictions/ranking

rank_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.learn_to_rank.make_rank_pipeline data/predictions/candidate_generation/features.csv \
	data/predictions/candidate_generation/playlists.csv data/predictions/candidate_generation/user_playlist.csv

diversify:
	$(PYTHON_INTERPRETER) -m src.models.diversifaction.diversify data/predictions/ranking \
	models/diversification data/predictions/diversification 

diversify_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.diversifaction.diversify_pipeline data/predictions/candidate_generation/features.csv \
	data/predictions/candidate_generation/playlists.csv data/predictions/candidate_generation/user_playlist.csv

coldstart:
	$(PYTHON_INTERPRETER) -m src.models.coldstart.cluster_coldstart data/processed models/coldstart

coldstart_birch:
	$(PYTHON_INTERPRETER) -m src.models.coldstart.cluster_birch_coldstart data/processed models/coldstart_birch

coldstart_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.coldstart.coldstart_pipeline models/coldstart/embedding.pkl models/coldstart/pca.pkl \
	models/coldstart/clusterer.pkl models/coldstart/clustered_tracks.pkl models/coldstart/songs_encodings.pkl

coldstart_birch_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.coldstart.coldstart_birch_pipeline models/coldstart_birch/embeddings_dict.pkl \
	models/coldstart_birch/pca.pkl models/coldstart_birch/brc.pkl models/coldstart_birch/clustered_tracks.pkl

rerank:
	$(PYTHON_INTERPRETER) -m src.models.reranking.rerank data/processed models/reranking

rerank_pipeline:
	$(PYTHON_INTERPRETER) -m src.models.reranking.rerank_pipeline \
	models/reranking/uris_dict.pkl \
	models/reranking/desired_distribution.pkl \


recommendation_pipeline:
	$(PYTHON_INTERPRETER) -m src.pipeline.recommendation_pipeline \
	models/candidate_generation/candidate_generator.pkl \
	data/dataset/dataset_lightfm \
	data/processed/songs_encodings.csv \
	data/processed/songs_features.csv \
	data/processed/playlists.csv

test_recommendation_pipeline:
	$(PYTHON_INTERPRETER) -m tests.models.rank_and_diversify.test_recommendation_pipeline \
	models/candidate_generation/candidate_generator.pkl \
	data/dataset/dataset_lightfm \
	data/processed/songs_encodings.csv \
	data/processed/songs_features.csv \
	data/processed/playlists.csv \
	data/processed/test_playlists.csv \
	reports/eval_rank_and_diversify

test_reranked_pipeline:
	$(PYTHON_INTERPRETER) -m tests.models.reranking.test_reranking \
	models/candidate_generation/candidate_generator.pkl \
	data/dataset/dataset_lightfm \
	data/processed/songs_encodings.csv \
	data/processed/songs_features.csv \
	data/processed/playlists.csv \
	data/processed/test_playlists.csv \
	models/reranking/uris_dict.pkl \
	models/reranking/desired_distribution.pkl \
	reports/eval_rank_and_diversify \
	models/functions

copy_artifacts:
	cp models/candidate_generation/candidate_generator.pkl serving/artifacts
	cp data/dataset/dataset_lightfm serving/artifacts
	cp data/processed/songs_encodings.csv serving/artifacts
	cp data/processed/songs_features.csv serving/artifacts
	cp data/processed/playlists.csv serving/artifacts
	cp models/reranking/uris_dict.pkl serving/artifacts
	cp models/reranking/desired_distribution.pkl serving/artifacts
	cp models/coldstart_birch/brc.pkl serving/artifacts
	cp models/coldstart_birch/clustered_tracks.pkl serving/artifacts
	cp models/coldstart_birch/embeddings_dict.pkl serving/artifacts
	cp models/coldstart_birch/pca.pkl serving/artifacts
	cp models/functions/candidate_generation_component.pkl serving/artifacts
	cp models/functions/prepare_test_playlist.pkl serving/artifacts
	cp models/functions/load_songs_encodings.pkl serving/artifacts
	cp models/functions/rank_and_diversify_component.pkl serving/artifacts
	cp models/functions/recommendation_reranking_component.pkl serving/artifacts

api:
	uvicorn -m serving.api:app --host 0.0.0.0 --port 80


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
