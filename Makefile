.PHONY: test install-user install-global install-vagrant install-deps

test:
	@cd k3d; python -m unittest discover -t test

install-user: install-deps
	@pip install --user .

install-global: install-deps
	@sudo pip install .

install-vagrant:
	@vagrant ssh -c "cd /vagrant; make install-user"

install-deps:
	@bower install --config.interactive=false
