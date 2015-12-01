# pip
define jupyter::pip($package = $title, $ensure = installed, $timeout=300) {
    $dir = sprintf(
        '/usr/local/lib/python2.?/dist-packages/%s-*-info',
        regsubst($package, '-', '_', 'G')
    )

    exec { "pip:${title}":
        command => "pip install ${package}",
        onlyif  => "/usr/bin/test ! -d ${dir}",
        timeout => $timeout,
        require => [Package['python-pip'], Package['python-dev']],
    }
}
