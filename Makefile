.PHONY: test install install-vagrant

test:
	@cd k3d; python -m unittest discover -t test

install:
	@bower install --config.interactive=false
	@sudo pip install .

install-vagrant:
	@vagrant ssh -c "cd /vagrant; make install"
