# jupyter
class jupyter ($user = 'vagrant', $directory = '/home/vagrant') {
    file { '/etc/profile.d/locale.sh':
        ensure  => present,
        content => 'export LC_ALL=en_US.utf8',
    }

    package { ['git', 'python-dev', 'python-pip', 'python3-dev', 'python3-pip']:
        ensure => installed,
    }

    pip { ['jupyter', 'jupyter-pip', 'numpy', 'pytest']:
        ensure => installed,
    }
    ->
    exec { 'install python2 kernel':
        command => '/usr/local/bin/ipython2 kernelspec install-self',
        creates => '/usr/local/share/jupyter/kernels/python2',
    }

    pip3 { ['jupyter', 'jupyter-pip', 'numpy', 'pytest']:
        ensure => installed,
    }
    ->
    exec { 'install python3 kernel':
        command => '/usr/local/bin/ipython3 kernelspec install-self',
        creates => '/usr/local/share/jupyter/kernels/python3',
    }

    package { 'supervisor':
        ensure => installed,
    }
    ->
    file { '/etc/supervisor/conf.d/jupyter.conf':
        ensure  => present,
        content => template('jupyter/jupyter.conf.erb'),
        notify  => Service['supervisor'],
    }
    ->
    service { 'supervisor':
        ensure => running,
        enable => true,
    }
}
