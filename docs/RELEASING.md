# Release checklist

1. **Credits.** Check the credits and permissions of every file in the packages (the CBBE 3BA body in the corset
   included), and credit every asset source in the README and the release notes.
2. **Version.** Set `$Version` in `build.ps1` and `presets\build.ps1`, and add a section to `CHANGELOG.md` and
   `presets\CHANGELOG.md`.
3. **Build.** Run `.\build.ps1` and `.\presets\build.ps1`; the packages land in `dist\` as
   `Hollow Lantern - CBBE 3BA (<res>) - <version>.7z` and `Hollow Lantern Body Presets - CBBE 3BA - <version>.7z`.
4. **Check the packages.**
   - `7z t` passes on every archive.
   - The outfit archives hold the plugin, `meshes`, `textures`, `CalienteTools`, `Scripts` and `Source` at the root
     (no `Data` folder level), and the `.pex` header carries no user or computer name; the presets archive holds only `CalienteTools\BodySlide\SliderPresets\Hollow Lantern Presets.xml`.
   - The three outfit packages share the same plugin, meshes and BodySlide files; only the textures differ.
   - DDS headers: BC7 with full mip chains; diffuse and normal map 2048 px (2K), 4096 px (4K) and 8192 px (8K), glow
     map and reflection mask a quarter of that.
   - Install a package in Mod Organizer 2, run BodySlide Batch Build for the group Hollow Lantern and test in game:
     every item in third and first person, the tail physics, the axe enchantment, the recipes (the lantern turns
     into the carried light when crafted).
5. **Release assets.** Copy the tested packages to `dist\` as `Hollow-Lantern-CBBE-3BA-<res>-<version>.7z` and
   `Hollow-Lantern-Body-Presets-CBBE-3BA-<version>.7z` (GitHub turns spaces and brackets in asset names into dots),
   and note their SHA-256 (`Get-FileHash`).
6. **Release.** Tag `v<version>` on `main` with installation notes and the checksums:

   ```powershell
   gh release create v1.0 --target main --title "Hollow Lantern 1.0" --notes-file notes.md `
       "dist\Hollow-Lantern-CBBE-3BA-2K-1.0.7z#Hollow Lantern - CBBE 3BA (2K) - 1.0" `
       "dist\Hollow-Lantern-CBBE-3BA-4K-1.0.7z#Hollow Lantern - CBBE 3BA (4K) - 1.0" `
       "dist\Hollow-Lantern-CBBE-3BA-8K-1.0.7z#Hollow Lantern - CBBE 3BA (8K) - 1.0" `
       "dist\Hollow-Lantern-Body-Presets-CBBE-3BA-1.0.7z#Hollow Lantern Body Presets - CBBE 3BA - 1.0"
   ```

7. **README.** Point the Downloads table at the new assets
   (`https://github.com/71Kevin/hollow-lantern-se/releases/download/v<version>/<file>`) and keep the Mod pages table up
   to date.
8. **Verify.** Download every asset from the release page and compare its SHA-256 with the tested package.
