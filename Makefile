all:
	make install-keras
	make install-pytorch

install-keras:
	export FRAMEWORK="keras" && pip install -e . --upgrade

install-pytorch:
	export FRAMEWORK="pytorch" && pip install -e . --upgrade

clean:
	rm -rf *egg-info build dist nbeats_pytorch/__pycache__ __pycache__ nbeats_keras/__pycache__ results .DS_Store .ipynb_checkpoints

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

run-jupyter:
	pip install jupyter
	jupyter notebook examples/NBeats.ipynb

test:
	pip install pytest
	pip install -r examples/examples-requirements.txt
	cd examples && python trainer_keras.py --task dummy --test
	cd examples && python trainer_pytorch.py --task dummy --test
	pytest
