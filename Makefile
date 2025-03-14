IMAGE_NAME=project_image

# Phony Targets
.PHONY: build interactive notebook flask \
	db_create db_load db_rm db_clean db_interactive

# Build our Docker image
build:
	docker build . -t $(IMAGE_NAME)

# Run container interactively. All files will be mounted except requirements
interactive: build
	docker run -it \
	-v "$(shell pwd):/app/src" \
	$(IMAGE_NAME) /bin/bash