all: install clean

install:
	pip install .

clean:
	rm -rf build pongrid.egg-info
