init:
	pip install -r requirements.txt

test:
	nosetests -v tests

freeze_package:
	poetry export --without-hashes -f requirements.txt --output requirements.txt

test-pytest:
	pytest tests/ --log-cli-level=warning --cov=./ --cov-report term-missing