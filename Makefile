COMPOSE_CMD	= docker compose
USER_ID		= $(shell id -u)
USER_NAME	= $(shell id -u --name)
GROUP_ID	= $(shell id -u)
CURRENT_DIR	= $(shell pwd)
DATASET_DIR	= /storage
DOCKER_DIR	= docker/
BUILD_ARGS	= --build-arg USER_ID=$(USER_ID) --build-arg USER_NAME=$(USER_NAME) --build-arg GROUP_ID=$(GROUP_ID)


run:
	jupyter lab --ip 0.0.0.0 --no-browser --allow-root

experiment:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-experiments.yml up

run-jupyter:
	docker run -v $(DATASET_DIR):/storage -v $(CURRENT_DIR):/app --gpus all -it --rm -p 8888:8888 main-jupyter:latest

run-compose-jupyter:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml up

run-maggie:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml up

build:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-experiments.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

build-compose-jupyter:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

build-maggie:
	mv pretrained_models ../.
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml build $(BUILD_ARGS)
	mv ../pretrained_models .

build-jupyter:
	mv pretrained_models ../.
	docker build -t main-jupyter:latest -f $(DOCKER_DIR)Dockerfile-jupyter $(BUILD_ARGS) .
	mv ../pretrained_models .

down:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-jupyter.yml down

down-maggie:
	$(COMPOSE_CMD) -f $(DOCKER_DIR)docker-compose-maggie.yml down

clean:
	rm -rf pretrained_models/**/training-logs/checkpoints/

