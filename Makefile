all:
	make install-keras
	make install-pytorch

install-keras:
	export FRAMEWORK="keras" && pip install . --upgrade

install-pytorch:
	export FRAMEWORK="pytorch" && pip install . --upgrade

clean:
	rm -rf *egg-info

