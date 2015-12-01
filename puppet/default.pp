node default {
    include [jupyter, bower]
}

Exec { path => ['/bin', '/usr/bin', '/usr/local/bin'] }

exec { 'apt-get update': }

Exec['apt-get update'] -> Package <| |>
