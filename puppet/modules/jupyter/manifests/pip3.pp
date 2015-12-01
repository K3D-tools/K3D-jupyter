# pip3
define jupyter::pip3($package = $title, $ensure = installed, $timeout=300) {
    $dir = sprintf(
        '/usr/local/lib/python3.?/dist-packages/%s-*-info',
        regsubst($package, '-', '_', 'G')
    )

    exec { "pip3:${title}":
        command => "pip3 install ${package}",
        onlyif  => "/usr/bin/test ! -d ${dir}",
        timeout => $timeout,
        require => [Package['python3-pip'], Package['python3-dev']],
    }
}
