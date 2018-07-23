'use strict';

var express = require('express'),
    os = require('os'),
    app = express(),
    cors = require('cors'),
    fs = require('fs'),
    platform,
    server;

if (os.platform() === 'win32') {
    platform = 'win32';
} else {
    platform = 'linux';
}

app.use(cors({
    credentials: true,
    origin: true,
    exposedHeaders: ['Location']
}));

app.use(function (req, res, next) {
    var data = '';

    req.setEncoding('utf8');
    req.on('data', function (chunk) {
        data += chunk;
    });
    req.on('end', function () {
        req.rawBody = data;
        next();
    });
});

app.post('/screenshots/:filename', function (req, res) {
    console.log(req.rawBody);

    fs.writeFile('test/results/' + req.params.filename, req.rawBody, 'base64', function () {
        res.writeHead(200);
        res.end();
    });
});

app.get('/screenshots/:filename', function (req, res) {
    var path = 'test/reference/' + req.params.filename;

    if (!fs.existsSync(path)) {
        path = 'test/reference/' + platform + '/' + req.params.filename;
        if (!fs.existsSync(path)) {
            res.status(404).send('Sorry cant find that!');
            return;
        }
    }

    fs.readFile(path, 'base64', function (err, data) {
        res.end(data);
    });
});

app.get('/samples/:filename', function (req, res) {
    fs.readFile('test/samples/' + req.params.filename, function (err, data) {
        res.end(data);
    });
});

server = app.listen(9001, function () {
    console.log('App listening at port %s', server.address().port);
});
