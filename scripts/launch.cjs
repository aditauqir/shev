const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const os = require('node:os');
const packageName = require('../package.json').name;

module.exports = function launch(binary) {
  const root = path.resolve(__dirname, '..');
  const native = path.join(root, 'native', 'bin', binary);
  if (!fs.existsSync(native)) {
    console.error(`shev's native binary is missing. Run npm rebuild -g ${packageName} --foreground-scripts.`);
    process.exit(1);
  }
  const args = process.argv.slice(2);
  const state = path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share'), 'shev-npm', 'state');
  const marker = path.join(state, '.shev-owned');
  try {
    if (!fs.existsSync(state)) {
      fs.mkdirSync(state, { recursive: true, mode: 0o700 });
      fs.writeFileSync(marker, 'shev managed state\n', { flag: 'wx', mode: 0o600 });
    }
    if (fs.readFileSync(marker, 'utf8') !== 'shev managed state\n') throw new Error('unrecognized state directory');
    const installId = fs.readFileSync(path.join(root, '.shev-install-id'), 'utf8');
    const manifestPath = path.join(state, 'config', 'install.json');
    const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, 'utf8')) : {};
    if (manifest.installation_id !== installId) {
      const registration = spawnSync(native, ['--register-install', path.dirname(native)], {
        encoding: 'utf8', env: { ...process.env, SHEV_HOME: state, SHEV_INSTALL_ID: installId },
      });
      if (registration.status !== 0) throw new Error(registration.stderr || 'could not register installation ownership');
    }
  } catch (error) {
    console.error(`Could not initialize shev safely: ${error.message}`);
    process.exit(1);
  }
  const result = spawnSync(native, args, {
    stdio: 'inherit',
    env: { ...process.env, SHEV_HOME: state },
  });
  if (result.error) console.error(result.error.message);
  if (result.status !== 0 || args[0] !== '--uninstall' || args.includes('--dry-run')) {
    process.exit(result.status ?? 1);
  }
  // npm >=7 intentionally does not run uninstall lifecycle hooks. The explicit
  // native uninstaller cleans external registrations first; only then remove npm.
  const npmRoot = spawnSync('npm', ['root', '--global'], { encoding: 'utf8' });
  const sameRoot = (candidate) => {
    try { return fs.realpathSync(candidate) === fs.realpathSync(root); } catch { return false; }
  };
  let uninstallArgs;
  if (npmRoot.status === 0 && sameRoot(path.join(npmRoot.stdout.trim(), packageName))) {
    uninstallArgs = ['uninstall', '--global', packageName];
  } else {
    const modules = path.dirname(path.dirname(root));
    if (path.basename(modules) === 'node_modules') {
      uninstallArgs = ['uninstall', '--prefix', path.dirname(modules), packageName];
    }
  }
  if (uninstallArgs) {
    const removal = spawnSync('npm', uninstallArgs, { stdio: 'inherit' });
    process.exit(removal.status ?? 1);
  }
  console.log('Native shev installation removed. Source checkout kept.');
};
