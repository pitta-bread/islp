.PHONY: lint run

lint:
	uv run ty check
	uv run ruff check

CHAPTER ?= 1

run:
	uv run islp --chapter $(CHAPTER)
