.PHONY: test install-user install-global install-vagrant install-deps

PIP := 'pip'

test:
	@vagrant ssh -c "cd /vagrant; python2 -m pytest"
	@vagrant ssh -c "cd /vagrant; python3 -m pytest"

install-user: install-deps
	@$(PIP) install --user .

install-global: install-deps
	@sudo $(PIP) install .

install-vagrant:
	@vagrant ssh -c "cd /vagrant; make install-user"
	@vagrant ssh -c "cd /vagrant; make install-user PIP=pip3"

install-deps:
	@bower install --config.interactive=false
