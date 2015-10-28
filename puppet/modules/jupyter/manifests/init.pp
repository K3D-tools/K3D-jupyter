class jupyter ($user = 'vagrant', $directory = '/home/vagrant') {
    file { '/etc/profile.d/locale.sh':
        content => 'export LC_ALL=en_US.utf8',
        ensure => present
    }

    package { ['git', 'python-dev', 'python-pip']:
        ensure => installed,
    }
    ->
    package { ['jupyter', 'jupyter-pip']:
        provider => pip,
        ensure => installed,
    }

    package { 'supervisor':
        ensure => installed,
    }
    ->
    file { '/etc/supervisor/conf.d/jupyter.conf':
        content => template('jupyter/jupyter.conf.erb'),
        ensure => present,
        notify => Service['supervisor'],
    }
    ->
    service { 'supervisor':
        ensure => running,
        enable => true,
    }
}
