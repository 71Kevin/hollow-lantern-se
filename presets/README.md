# Hollow Lantern Body Presets - CBBE 3BA

Five BodySlide presets for **CBBE 3BA** by **Tinesh**, made to wear the
[Hollow Lantern](../README.md) outfit. They work with any CBBE 3BA outfit. They are a separate download: the outfit
does not need them and they do not need the outfit.

**Download:** `Hollow-Lantern-Body-Presets-CBBE-3BA-1.0.7z` from the
[latest release](https://github.com/71Kevin/hollow-lantern-se/releases/latest), or as an optional file on the
[Nexus Mods](https://www.nexusmods.com/skyrimspecialedition/mods/194674?tab=files) and
[Dwemer Mods](https://dwemermods.com/mods/4993) pages.

## Presets

| Preset | Shape |
|---|---|
| Hollow Lantern - Harvest Queen | Voluptuous: very full bust, soft waist, wide hips, heavy thighs and a round butt |
| Hollow Lantern - Tavern Witch | Soft curves: full bust that grows with weight, rounded belly and hips |
| Hollow Lantern - Night Huntress | Athletic: defined abs, high round butt, strong thighs and calves |
| Hollow Lantern - Wisp Dancer | Full bust over a narrow waist, wide hips and toned legs |
| Hollow Lantern - Ember Warden | Strong and muscular: broad shoulders, defined abs and arms; lean at low weight |

Low weight is a lighter version of the same body (Harvest Queen, Night Huntress and Wisp Dancer at about 82–85% of
the volume; Tavern Witch and Ember Warden have their own low-weight shapes).

## Made to wear clothes

Curvy presets often push the body through itself: the breasts pass through each other at the cleavage and the inner
thighs overlap at the crotch. Garments copy that, and the result is clipping between the breasts and between the
legs. These presets keep the curves but not those overlaps:

- the breasts never cross the middle of the cleavage and the inner thighs never overlap (no crossing vertices at
  either weight on all five presets);
- fewer folds in the body itself between the breasts and under the bust.

## Requirements

- [CBBE 3BA (3BBB)](https://www.nexusmods.com/skyrimspecialedition/mods/30174).
- [BodySlide and Outfit Studio](https://www.nexusmods.com/skyrimspecialedition/mods/201).

## Installation

1. Install the archive with your mod manager (Mod Organizer 2 or Vortex), or extract it into the game's `Data`
   folder. It only adds `CalienteTools\BodySlide\SliderPresets\Hollow Lantern Presets.xml`.
2. In BodySlide, pick the preset (it is listed under the groups CBBE, 3BA, 3BBB, CBBE Bodies and Hollow Lantern),
   build the body, then run **Batch Build** for your outfits with **Build Morphs** ticked.

Nothing to enable in the load order: there is no plugin.

## How they were made

Each preset was fitted by a script on the CBBE 3BA reference body: a target silhouette is reproduced with a chosen
set of shape sliders (no detail sliders for nipples or genitals), with the two no-overlap rules above enforced
during the fit, then rounded to whole slider values.

## Building from source

`build.ps1` regenerates the XML with Blender and writes `Hollow Lantern Body Presets - CBBE 3BA - <version>.7z` to
`dist\`. It uses the CBBE 3BA reference body extracted by the outfit build (run `..\build.ps1` first) and reads its
target presets from a Mod Organizer 2 mods folder (`-Mods`). With `-Install` it also installs the package into the
Mod Organizer 2 mod folder (MO2 closed).

## Credits

- Presets: Tinesh.

## Support My Work

If you enjoy my mods and want to support future projects, you can buy me a coffee on Ko-fi:
[**Support me on Ko-fi**](https://ko-fi.com/tinesh)
