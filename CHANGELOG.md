# Changelog

## 1.0 — 2026-10-09

First version, in three texture packages: 2K (2048 px atlas), 4K (4096 px) and 8K (8192 px).

### Added
- Hollow Lantern Corset (body slot): soot-black leather corset with stitched boning channels, front lacing over an
  orange wool modesty panel, brass eyelets and attached briefs; the corset carries the CBBE 3BA body with the skin
  under the garment trimmed away.
- Hollow Lantern Shorts (slot 49): fitted burnt-orange wool shorts with leather leg cuffs, a waistband and a belt with
  a brass buckle.
- Hollow Lantern Boots (slot 37): over-knee witch boots with a Louis heel (high-heels offset 4.8), front lacing over
  an orange tongue and brass heel tips.
- Hollow Lantern Gloves (slot 33): leather opera gloves with a brass button; own first-person model.
- Hollow Lantern Choker (slot 45): leather choker with a brass ring on a leather loop and a small pumpkin charm.
- Hollow Lantern Horns (slot 42): curved horns fading to an ember tip, with brass cuffs.
- Hollow Lantern Tail (slot 40): leather tail with a spade tip, SMP physics on ten bones, colliding with an invisible
  body proxy.
- Hollow Lantern Mask (slot 44): a black leather masquerade half-mask with a widow's peak, orange stitching, brass
  eye frames, glowing orange pumpkin-vine inlays and a brass pumpkin medallion with an amber stone; it leaves the
  mouth and chin free.
- Hollow Lantern War Axe and Hollow Lantern Battleaxe: lantern axes in black and orange — a bearded crescent blade of
  blackened steel with a glowing ember edge, three gothic lantern windows of amber glass framed in brass, orange
  enamel vines and a caged pumpkin lantern on top; a black haft laced with orange cord. The war axe has a hook on the
  back, the battleaxe a second, smaller blade. Damage 14 and 24.
- Hollow Lantern Ember enchantment, stronger on the battleaxe: fire damage on hit (18 / 26), a burn that does not
  stack (4 / 6 per second for 4 seconds), a 10-second soul trap and Lantern's Dread, which makes wounded targets
  (below 35 % health, up to level 35 / 45) flee for 8 / 10 seconds.
- Hollow Lantern (light): a carved jack-o'-lantern on a corded bail, carried in the left hand like a torch, with a
  flickering candle, flame and halo effects and a warm dynamic light that never burns out.
- BodySlide group "Hollow Lantern" with one project per body piece (all CBBE 3BA sliders, Build Morphs supported).
- Forge recipes for every item (no perk needed), tempering for the corset, boots and gloves at the workbench and for
  the axes at the grindstone.

### Fixed during testing (before release)
- The Hollow Lantern can be crafted at the forge and found by item spawners: crafting menus cannot list light
  sources, so the recipe makes a lantern item that a small script turns into the carried light as soon as it reaches
  an inventory or container.
- Equipping the tail no longer collapses the upper body: its collision shapes only use body bones, all declared as
  fixed in the physics file.
- Inventory previews, dropped items and the carried lantern are visible (static meshes no longer carry the skinned
  shader flag).
- The cups keep their full round shape on large-breast presets, and the skin near the neckline no longer pokes
  through the leather at the inner top of the cups (checked against 108 BodySlide presets).
- The briefs no longer follow the inside of the 3BA vulva and they bridge the butt cleft.
- The shorts no longer crumple at the crotch on curvy presets: the crotch is a smooth piece between the thighs with
  a little more room, and the shorts and briefs take the weights of the skin directly under them, so the thighs no
  longer pass through the shorts when walking.
- No more dark gaps in the skin just above the corset's top edge under the right arm: the trimmed skin under the
  corset now always ends below the leather.
- The corset edges and the choker follow the skin under them exactly, so breast physics and head movement no longer
  push skin through them; the choker follows the head where it sits on the neck of the head mesh.
- The choker's ring and charm hang from a leather loop on the band instead of floating in front of it, and the
  charm follows the skin of the throat, so it no longer sinks into the neck when the head nods or turns.
- Brass, cords, edges, linings, soles and the lantern's candle and flesh show their intended colours (their texture
  coordinates pointed at the wrong part of the atlas).
- No stretched texture at the front of the choker, waistband and belt or along the glove sleeves.
