clean:
	-rm -rf build
	-rm -rf dist
	-rm -rf *.egg-info
	-rm -rf .coverage
	-rm -rf cache
	-rm -rf junit
	-rm -rf joblib
	-rm -rf __pycache__
	-rm -rf .mypy_cache
	# -rm -rf .pytest_cache

test: clean
	pytest wsi \
		--doctest-modules \
		--junitxml=junit/test-results.xml \
		--cov=wsi \
		--cov-report=xml \
		--cov-report=html

install: clean
	python -m pip install -e .
