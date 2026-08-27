// Runs from `npm publish`, after the build. The "files" allowlist in package.json is the real
// defence against shipping stray directories - 2.18.0 went out with 38 MB of .yarn/cache because
// .npmignore, being present, switches npm off .gitignore and only listed "test". This guards what
// an allowlist cannot: its own removal, and a dist/ that was never rebuilt.
const fs = require('fs');
const path = require('path');

const BUNDLES = ['standalone.js', 'standalone.js.map'];
const dist = path.join(__dirname, 'dist');

function fail(message) {
    console.error(`prepublish: ${message}`);
    process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(path.join(__dirname, 'package.json'), 'utf8'));

if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    fail('package.json has no "files" allowlist - without it npm publishes every untracked '
        + 'directory in js/, including .yarn/cache');
}

if (!manifest.license) {
    fail('package.json has no "license" - the registry would show the package as unlicensed');
}

const missing = BUNDLES.filter((name) => !fs.existsSync(path.join(dist, name)));

if (missing.length > 0) {
    fail(`dist/ is missing ${missing.join(', ')} - run npm run build`);
}

function newestMtime(dir) {
    return fs.readdirSync(dir, { withFileTypes: true }).reduce((newest, entry) => {
        const full = path.join(dir, entry.name);
        const mtime = entry.isDirectory() ? newestMtime(full) : fs.statSync(full).mtimeMs;

        return Math.max(newest, mtime);
    }, 0);
}

const sources = newestMtime(path.join(__dirname, 'src'));
const built = Math.min(...BUNDLES.map((name) => fs.statSync(path.join(dist, name)).mtimeMs));

if (built < sources) {
    fail('dist/ is older than src/ - run npm run build');
}

console.log('prepublish: dist/ complete and newer than src/, allowlist and license in place');
