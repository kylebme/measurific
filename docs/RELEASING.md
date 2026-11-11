# Releasing

Releases are built by GitHub Actions
([.github/workflows/desktop.yml](../.github/workflows/desktop.yml)) on macOS
(arm64), Windows (x64) and Linux (x64), and published to
[GitHub Releases](https://github.com/kylebme/measurific/releases). The app's
auto-updater (`electron-updater`) reads the published release.

## Cut a release

1. Make sure `main` is green: open a pull request, or run the **Desktop app**
   workflow manually (**Actions → Desktop app → Run workflow**). Each run tests
   everything, builds all three installers, smoke-tests the packaged worker and
   camera bridge, and uploads the installers as workflow artifacts.
2. Bump the version in [app/package.json](../app/package.json) (and
   `app/package-lock.json`):

   ```bash
   cd app && npm version 0.2.0 --no-git-tag-version
   ```

3. Commit, then tag the commit with the **same** version, prefixed with `v`, and
   push the tag:

   ```bash
   git commit -am "Release 0.2.0"
   git tag v0.2.0
   git push origin main v0.2.0
   ```

   The workflow fails if the tag and `package.json` version differ.
4. When the workflow finishes, a **draft** release named `0.2.0` contains the
   `.dmg`, `.zip`, `.exe`, `.AppImage`, block maps and `latest*.yml` update
   files. Edit the notes (**Generate release notes** works well) and click
   **Publish release**. Installed apps see the update only once it is
   published.

Keep the `latest*.yml` and `.blockmap` files attached: the auto-updater needs
them. The macOS `.zip` exists for updates; users download the `.dmg`.

## Build locally

```bash
cd app
npm ci
npm run dist:mac      # or dist:linux, or dist:win (needs MSYS2 for the camera bridge)
node scripts/smoke-worker.mjs --resources "release/mac-arm64/Measurific.app/Contents/Resources"
```

Installers land in `app/release/`. Local builds never publish.

## Code signing

Builds are not signed with a paid certificate yet. macOS builds are ad-hoc
signed with hardened runtime. To sign and notarize, replace `"identity": "-"`
in `app/package.json` with a Developer ID, add notarization, and provide the
certificates to the workflow as repository secrets (`CSC_LINK`,
`CSC_KEY_PASSWORD`, `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID`).
Windows signing uses `WIN_CSC_LINK` / `WIN_CSC_KEY_PASSWORD`.
