all:
	make install-keras
	make install-pytorch

install-keras:
	export FRAMEWORK="keras" && pip install . --upgrade

install-pytorch:
	export FRAMEWORK="pytorch" && pip install . --upgrade

clean:
	rm -rf *egg-info build dist nbeats_pytorch/__pycache__ __pycache__ nbeats_keras/__pycache__ results .DS_Store

deploy-keras:
	pip install twine --upgrade
	make clean
	export FRAMEWORK="keras" && python setup.py sdist bdist_wheel
	twine upload dist/*

deploy-pytorch:
	pip install twine --upgrade
	make clean
	export FRAMEWORK="pytorch" && python setup.py sdist bdist_wheel
	twine upload dist/*

deploy:
	make deploy-keras
	make deploy-pytorch

