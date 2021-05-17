install:
	pip install -r requirements.txt

build:
	python -m build

clean_build:
	rm -rf build/ && rm -rf dist/ && rm -rf src/KeyExtractor.egg-info/

upload_test:
	twine upload -r testpypi dist/*

upload:
	twine upload dist/*

build_and_upload_test:
	$(MAKE) clean_build && \
	$(MAKE) build && \
	$(MAKE) upload_test

build_and_upload:
	$(MAKE) clean_build && \
	$(MAKE) build && \
	$(MAKE) upload

pip_download_test:
	python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps KeyExtractor

pip_install:
	pip install KeyExtractor

test:
	nosetests -v tests

freeze_package:
	poetry export --without-hashes -f requirements.txt --output requirements.txt

test-pytest:
	pytest tests/ --log-cli-level=warning --cov=./ --cov-report term-missing