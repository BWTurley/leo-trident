# macOS Deployment

These are LaunchAgent templates for running Leo Trident on a Mac Mini.
They are not executed by any CI or install script — the user copies them
into `~/Library/LaunchAgents/` during manual setup.

## Files
- `com.leotrident.consolidate.plist` — nightly 2 AM consolidation
- `com.leotrident.watcher.plist` — always-on vault file watcher
- `com.leotrident.backup.plist` — weekly Sunday 3 AM SQLite + LanceDB backup
- `com.leotrident.health.plist` — always-on health endpoint on 127.0.0.1:8765

## Setup
1. Replace every `REPLACE_USERNAME` with the output of `whoami`.
2. Replace `REPLACE_HOME` with `$HOME` (e.g. `/Users/brett`).
3. Copy both plists to `~/Library/LaunchAgents/`.
4. `launchctl load ~/Library/LaunchAgents/com.leotrident.*.plist`
5. `launchctl list | grep leotrident` to verify both are loaded.

## Gotchas
- Do not place `LEO_TRIDENT_HOME` under iCloud Drive — the watcher will
  trigger on every iCloud sync event.
- If launchd silently fails, check `data/*.stderr.log`. 90% of the time
  it's a bad path in ProgramArguments.
- macOS may prompt for Full Disk Access the first time launchd touches
  the vault. Grant it to the venv python binary.
