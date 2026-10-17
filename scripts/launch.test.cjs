const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function fixture() {
  const home = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'shev-npm-test-')));
  const root = path.join(home, 'global', 'lib', 'node_modules', '@noorslvt', 'shev');
  for (const name of ['scripts', 'bin', 'native/bin']) fs.mkdirSync(path.join(root, name), { recursive: true });
  for (const name of ['scripts/launch.cjs', 'bin/shev.cjs', 'bin/shev-cli.cjs', 'package.json']) {
    fs.copyFileSync(path.join(__dirname, '..', name), path.join(root, name));
  }
  fs.writeFileSync(path.join(root, '.shev-install-id'), 'fixture-install');
  const native = `#!${process.execPath}\nconst fs=require('node:fs'),path=require('node:path');if(process.argv[2]==='--register-install'){const dir=path.join(process.env.SHEV_HOME,'config');fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'install.json'),JSON.stringify({installation_id:process.env.SHEV_INSTALL_ID}));process.exit(0);}console.log(JSON.stringify({args:process.argv.slice(2),home:process.env.SHEV_HOME})); process.exit(Number(process.env.FIXTURE_EXIT||0));\n`;
  for (const name of ['shev', 'shev-cli']) fs.writeFileSync(path.join(root, 'native/bin', name), native, { mode: 0o755 });
  const mockBin = path.join(home, 'mock-bin'); fs.mkdirSync(mockBin);
  const npmLog = path.join(home, 'npm-log.json');
  fs.writeFileSync(path.join(mockBin, 'npm'), `#!${process.execPath}\nconst fs=require('node:fs');const args=process.argv.slice(2);if(args[0]==='root')console.log(${JSON.stringify(path.join(home,'global/lib/node_modules'))});else fs.writeFileSync(${JSON.stringify(npmLog)},JSON.stringify(args));\n`, { mode: 0o755 });
  const env = { ...process.env, HOME: home, PATH: mockBin + path.delimiter + process.env.PATH };
  return { home, root, env, npmLog };
}

test('npm exposes shev and shev-cli, forwarding arguments and isolating its state', () => {
  const f = fixture();
  try {
    for (const name of ['shev', 'shev-cli']) {
      const result = spawnSync(process.execPath, [path.join(f.root, 'bin', `${name}.cjs`), '--file', 'a file.json'], { env: f.env, encoding: 'utf8' });
      assert.equal(result.status, 0, result.stderr);
      const output = JSON.parse(result.stdout);
      assert.deepEqual(output.args, ['--file', 'a file.json']);
      assert.equal(output.home, path.join(f.home, '.local/share/shev-npm/state'));
    }
  } finally { fs.rmSync(f.home, { recursive: true }); }
});

test('dry-run and cancelled removal never invoke npm uninstall', () => {
  const f = fixture();
  try {
    const wrapper = path.join(f.root, 'bin/shev.cjs');
    let result = spawnSync(process.execPath, [wrapper, '--uninstall', '--dry-run'], { env: f.env });
    assert.equal(result.status, 0); assert.equal(fs.existsSync(f.npmLog), false);
    result = spawnSync(process.execPath, [wrapper, '--uninstall'], { env: { ...f.env, FIXTURE_EXIT: '2' } });
    assert.equal(result.status, 2); assert.equal(fs.existsSync(f.npmLog), false);
  } finally { fs.rmSync(f.home, { recursive: true }); }
});

test('full confirmed removal invokes npm only for the package owning this wrapper', () => {
  const f = fixture();
  try {
    const result = spawnSync(process.execPath, [path.join(f.root, 'bin/shev.cjs'), '--uninstall', '--yes'], { env: f.env });
    assert.equal(result.status, 0);
    assert.deepEqual(JSON.parse(fs.readFileSync(f.npmLog)), ['uninstall', '--global', '@noorslvt/shev']);
  } finally { fs.rmSync(f.home, { recursive: true }); }
});
