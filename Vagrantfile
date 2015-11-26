# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|
    config.vm.box = "ubuntu/trusty64"
    config.vm.box_check_update = false

    config.vm.network "forwarded_port", guest: 8888, host: 8888
    config.vm.synced_folder "examples", "/home/vagrant/examples"

    config.vm.provision :puppet do |puppet|
        puppet.manifests_path = "puppet"
        puppet.module_path = "puppet/modules"
    end
end
