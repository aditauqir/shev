const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { randomUUID } = require('node:crypto');

const root = path.resolve(__dirname, '..');
const state = path.join(root, '.install-state');
const installId = randomUUID();
fs.writeFileSync(path.join(root, '.shev-install-id'), installId);
if (process.platform === 'win32') {
  console.error('shev currently supports macOS/Linux terminals; on Windows use WSL.');
  process.exit(1);
}
fs.mkdirSync(state, { recursive: true });
fs.writeFileSync(path.join(state, '.shev-owned'), 'shev managed state\n');
const result = spawnSync('bash', [path.join(root, 'install.sh')], {
  stdio: 'inherit',
  env: {
    ...process.env,
    SHEV_PREFIX: path.join(root, 'native'),
    SHEV_SOURCE_DIR: root,
    SHEV_HOME: state,
    SHEV_NO_PATH_SETUP: '1',
    SHEV_INSTALL_ID: installId,
  },
});
if (result.error) console.error(`Could not install shev: ${result.error.message}`);
process.exit(result.status ?? 1);
