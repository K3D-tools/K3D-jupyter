# pip
define jupyter::pip($package = $title, $ensure = installed) {
    $dir = sprintf(
        '/usr/local/lib/python2.?/dist-packages/%s-*-info',
        regsubst($package, '-', '_', 'G')
    )

    exec { "pip:${title}":
        command => "/usr/bin/pip install ${package}",
        onlyif  => "/usr/bin/test ! -d ${dir}",
        require => [Package['python-pip'], Package['python-dev']]
    }
}
