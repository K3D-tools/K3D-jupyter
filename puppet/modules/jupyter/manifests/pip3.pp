# pip3
define jupyter::pip3($package = $title, $ensure = installed) {
    $dir = sprintf(
        '/usr/local/lib/python3.?/dist-packages/%s-*-info',
        regsubst($package, '-', '_', 'G')
    )

    exec { "pip3:${title}":
        command => "/usr/bin/pip3 install ${package}",
        onlyif  => "/usr/bin/test ! -d ${dir}",
        require => [Package['python3-pip'], Package['python3-dev']]
    }
}
