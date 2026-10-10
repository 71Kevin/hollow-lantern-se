using System.Drawing;
using System.Text.Json;
using Mutagen.Bethesda;
using Mutagen.Bethesda.Plugins;
using Mutagen.Bethesda.Plugins.Records;
using Mutagen.Bethesda.Skyrim;
using Noggog;

var skyrimPath = args[0];
var outDir = args[1];
using var meshReport = JsonDocument.Parse(File.ReadAllText(args[2]));
var version = args[3];

ObjectBounds Bounds(JsonElement e)
{
    var o = e.GetProperty("obnd").EnumerateArray().Select(v => (short)v.GetInt32()).ToArray();
    return new ObjectBounds { First = new P3Int16(o[0], o[1], o[2]), Second = new P3Int16(o[3], o[4], o[5]) };
}
JsonElement Item(string key) => meshReport.RootElement.GetProperty("items").GetProperty(key);

var modKey = ModKey.FromFileName("[Tinesh] Hollow Lantern.esp");
FormKey Mine(uint id) => new(modKey, id);
const SkyrimRelease release = SkyrimRelease.SkyrimSE;
const string meshDir = @"Tinesh\HollowLantern\";

using var skyrim = SkyrimMod.CreateFromBinaryOverlay(skyrimPath, release);
T Find<T>(IEnumerable<T> records, string editorId) where T : IMajorRecordGetter =>
    records.FirstOrDefault(r => r.EditorID == editorId) ?? throw new InvalidOperationException($"{editorId} not found in Skyrim.esm");
IFormLinkGetter<IKeywordGetter> Kw(string editorId) => Find(skyrim.Keywords, editorId).ToLinkGetter();
FormKey Ingredient(string editorId) =>
    skyrim.MiscItems.Cast<IMajorRecordGetter>().Concat(skyrim.Ingredients).Concat(skyrim.Ingestibles).Concat(skyrim.SoulGems)
        .FirstOrDefault(r => r.EditorID == editorId)?.FormKey ?? throw new InvalidOperationException($"{editorId} not found in Skyrim.esm");

var mod = new SkyrimMod(modKey, release);
mod.ModHeader.Flags |= SkyrimModHeader.HeaderFlag.Small;
mod.ModHeader.Author = "Tinesh";
mod.ModHeader.Description = $"Hollow Lantern {version}: a Halloween lantern-witch outfit for CBBE 3BA with a masquerade mask and enchanted lantern axes, crafted at the forge.";

var forge = Kw("CraftingSmithingForge");
var armorTable = Kw("CraftingSmithingArmorTable");
var leather = Ingredient("Leather01");
var strips = Ingredient("LeatherStrips");
var iron = Ingredient("IngotIron");
var boneMeal = Ingredient("BoneMeal");
var linen = Ingredient("RuinsLinenPile01");
var gourd = Ingredient("FoodGourd");
var gold = Ingredient("IngotGold");
var thorax = Ingredient("FireflyThorax");
var arcaneBlacksmith = new FormKey(ModKey.FromFileName("Skyrim.esm"), 0x05218E);

var races = new uint[]
{
    0x013740, 0x08883A, 0x013741, 0x08883C, 0x097A3D, 0x013742, 0x08883D, 0x0F71DC, 0x000D53, 0x067CD8, 0x0A82BA,
    0x013743, 0x088840, 0x013744, 0x088844, 0x013745, 0x088845, 0x10760A, 0x013746, 0x088794, 0x013747, 0x0A82B9,
    0x013748, 0x088846, 0x013749, 0x088884,
};
var defaultRace = Find(skyrim.Races, "DefaultRace").FormKey;
var firstPersonBody = Find(skyrim.ArmorAddons, "NakedTorso").FirstPersonModel!.Female!.File.DataRelativePath.ToString()
    .Replace(@"meshes\", "", StringComparison.OrdinalIgnoreCase);
var clothingUp = Find(skyrim.SoundDescriptors, "ITMClothingUpSD").FormKey;
var clothingDown = Find(skyrim.SoundDescriptors, "ITMClothingDownSD").FormKey;

BipedObjectFlag Slot(int slot) => (BipedObjectFlag)(1u << (slot - 30));

var pieces = new[]
{
    new { Id = 0u, Key = "corset", Name = "Corset", Slot = 32, Type = ArmorType.LightArmor, Rating = 20f, Weight = 3f, Value = 180u,
          Mesh = "corset_1.nif", Sliders = true, FirstPerson = firstPersonBody, Sounds = "ArmorLeatherCuirass",
          Keywords = new[] { "ArmorLight", "ArmorCuirass", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (leather, 3), (strips, 2), (iron, 1) } },
    new { Id = 2u, Key = "boots", Name = "Boots", Slot = 37, Type = ArmorType.LightArmor, Rating = 7f, Weight = 1.5f, Value = 70u,
          Mesh = "boots_1.nif", Sliders = true, FirstPerson = "", Sounds = "ArmorLeatherBoots",
          Keywords = new[] { "ArmorLight", "ArmorBoots", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (leather, 2), (strips, 2), (iron, 1) } },
    new { Id = 3u, Key = "gloves", Name = "Gloves", Slot = 33, Type = ArmorType.LightArmor, Rating = 6f, Weight = 0.5f, Value = 45u,
          Mesh = "gloves_1.nif", Sliders = true, FirstPerson = meshDir + "gloves_1.nif", Sounds = "ArmorLeatherGauntlets",
          Keywords = new[] { "ArmorLight", "ArmorGauntlets", "ArmorMaterialLeather", "VendorItemArmor" },
          Recipe = new[] { (leather, 1), (strips, 1) } },
    new { Id = 4u, Key = "choker", Name = "Choker", Slot = 45, Type = ArmorType.Clothing, Rating = 0f, Weight = 0.2f, Value = 40u,
          Mesh = "choker_1.nif", Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorClothing", "ClothingNecklace", "VendorItemJewelry" },
          Recipe = new[] { (strips, 1), (iron, 1) } },
    new { Id = 5u, Key = "horns", Name = "Horns", Slot = 42, Type = ArmorType.Clothing, Rating = 0f, Weight = 0.5f, Value = 55u,
          Mesh = "horns.nif", Sliders = false, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorClothing", "ClothingCirclet", "VendorItemJewelry" },
          Recipe = new[] { (boneMeal, 2), (strips, 1), (iron, 1) } },
    new { Id = 6u, Key = "tail", Name = "Tail", Slot = 40, Type = ArmorType.Clothing, Rating = 0f, Weight = 0.5f, Value = 40u,
          Mesh = "tail_1.nif", Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorClothing", "VendorItemClothing" },
          Recipe = new[] { (leather, 1), (strips, 1) } },
    new { Id = 7u, Key = "shorts", Name = "Shorts", Slot = 49, Type = ArmorType.Clothing, Rating = 0f, Weight = 1f, Value = 55u,
          Mesh = "shorts_1.nif", Sliders = true, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorClothing", "VendorItemClothing" },
          Recipe = new[] { (leather, 1), (linen, 2), (iron, 1) } },
    new { Id = 8u, Key = "mask", Name = "Mask", Slot = 44, Type = ArmorType.Clothing, Rating = 0f, Weight = 0.5f, Value = 75u,
          Mesh = "mask.nif", Sliders = false, FirstPerson = "", Sounds = "",
          Keywords = new[] { "ArmorClothing", "VendorItemClothing" },
          Recipe = new[] { (leather, 1), (strips, 1), (gold, 1) } },
};
uint RecipeId(uint id) => id < 7 ? 0x820u + id : 0x828u + (id - 7);

ConstructibleObject Recipe(uint id, string editorId, FormKey created, IFormLinkGetter<IKeywordGetter> bench, IEnumerable<(FormKey, int)> items)
{
    var c = new ConstructibleObject(Mine(id), release)
    {
        EditorID = editorId,
        CreatedObjectCount = 1,
        Items = new ExtendedList<ContainerEntry>(items.Select(i => new ContainerEntry
        {
            Item = new ContainerItem { Item = new FormLink<IItemGetter>(i.Item1), Count = i.Item2 },
        })),
    };
    c.CreatedObject.SetTo(created);
    c.WorkbenchKeyword.SetTo(bench.FormKey);
    mod.ConstructibleObjects.Add(c);
    return c;
}

void Temper(uint id, string name, FormKey created, IFormLinkGetter<IKeywordGetter> bench, FormKey material)
{
    var temper = Recipe(id, $"TemperHollowLantern{name}", created, bench, new[] { (material, 1) });
    temper.Conditions.Add(new ConditionFloat
    {
        CompareOperator = CompareOperator.NotEqualTo, ComparisonValue = 1, Flags = Condition.Flag.OR,
        Data = new EPTemperingItemIsEnchantedConditionData(),
    });
    var arcane = new HasPerkConditionData();
    arcane.Perk.Link.SetTo(arcaneBlacksmith);
    temper.Conditions.Add(new ConditionFloat { CompareOperator = CompareOperator.EqualTo, ComparisonValue = 1, Data = arcane });
}

uint temperId = 0x830;
foreach (var p in pieces)
{
    var flags = Slot(p.Slot);
    var model = meshDir + p.Mesh;
    var addon = new ArmorAddon(Mine(0x810 + p.Id), release)
    {
        EditorID = $"HollowLantern{p.Name}AA",
        BodyTemplate = new BodyTemplate { FirstPersonFlags = flags, ArmorType = p.Type },
        Priority = new GenderedItem<byte>(0, 0),
        WeightSliderEnabled = new GenderedItem<bool>(false, p.Sliders),
        WorldModel = new GenderedItem<Model?>(null, new Model { File = model }),
        FirstPersonModel = new GenderedItem<Model?>(null, p.FirstPerson.Length > 0 ? new Model { File = p.FirstPerson } : null),
    };
    addon.Race.SetTo(defaultRace);
    foreach (var r in races)
        addon.AdditionalRaces.Add(new FormLink<IRaceGetter>(new FormKey(ModKey.FromFileName("Skyrim.esm"), r)));
    mod.ArmorAddons.Add(addon);

    var ground = meshDir + p.Mesh.Replace("_1.nif", ".nif").Replace(".nif", "_gnd.nif");
    var armor = new Armor(Mine(0x800 + p.Id), release)
    {
        EditorID = $"HollowLantern{p.Name}",
        Name = $"Hollow Lantern {p.Name}",
        ObjectBounds = Bounds(Item(p.Key).GetProperty("gnd").GetProperty("bounds")),
        BodyTemplate = new BodyTemplate { FirstPersonFlags = flags, ArmorType = p.Type },
        WorldModel = new GenderedItem<ArmorModel?>(new ArmorModel { Model = new Model { File = ground } }, null),
        ArmorRating = p.Rating,
        Weight = p.Weight,
        Value = p.Value,
        Keywords = new ExtendedList<IFormLinkGetter<IKeywordGetter>>(p.Keywords.Select(Kw)),
    };
    armor.Race.SetTo(defaultRace);
    if (p.Sounds.Length > 0)
    {
        var template = Find(skyrim.Armors, p.Sounds);
        armor.PickUpSound.SetTo(template.PickUpSound.FormKeyNullable);
        armor.PutDownSound.SetTo(template.PutDownSound.FormKeyNullable);
    }
    else if (p.Type == ArmorType.Clothing && !p.Keywords.Contains("VendorItemJewelry"))
    {
        armor.PickUpSound.SetTo(clothingUp);
        armor.PutDownSound.SetTo(clothingDown);
    }
    armor.Armature.Add(addon.ToLink());
    mod.Armors.Add(armor);

    Recipe(RecipeId(p.Id), $"RecipeHollowLantern{p.Name}", armor.FormKey, forge, p.Recipe);
    if (p.Type == ArmorType.LightArmor)
        Temper(temperId++, p.Name, armor.FormKey, armorTable, strips);
}

var lantern = Find(skyrim.Lights, "Torch01").Duplicate(Mine(0x840));
lantern.EditorID = "HollowLanternLight";
lantern.Name = "Hollow Lantern";
lantern.Model = new Model { File = meshDir + "lantern.nif" };
lantern.ObjectBounds = Bounds(Item("lantern").GetProperty("bounds"));
lantern.Radius = 420;
lantern.Color = Color.FromArgb(0, 255, 158, 82);
lantern.Time = 10000000;
lantern.FadeValue = 1.2f;
lantern.Sound.Clear();
lantern.Weight = 1.5f;
lantern.Value = 35;
mod.Lights.Add(lantern);
var lanternToken = new MiscItem(Mine(0x850), release)
{
    EditorID = "HollowLanternToken",
    Name = "Hollow Lantern",
    Model = new Model { File = meshDir + "lantern.nif" },
    ObjectBounds = lantern.ObjectBounds,
    Weight = lantern.Weight,
    Value = lantern.Value,
};
lanternToken.VirtualMachineAdapter = new VirtualMachineAdapter
{
    Scripts =
    {
        new ScriptEntry
        {
            Name = "HollowLanternCraftScript",
            Flags = ScriptEntry.Flag.Local,
            Properties =
            {
                new ScriptObjectProperty { Name = "HollowLantern", Flags = ScriptProperty.Flag.Edited, Object = lantern.ToLink<ISkyrimMajorRecordGetter>() },
                new ScriptObjectProperty { Name = "HollowLanternToken", Flags = ScriptProperty.Flag.Edited, Object = lanternToken.ToLink<ISkyrimMajorRecordGetter>() },
            },
        },
    },
};
mod.MiscItems.Add(lanternToken);
Recipe(0x827, "RecipeHollowLanternLight", lanternToken.FormKey, forge, new[] { (gourd, 1), (thorax, 1), (iron, 1) });

var steel = Ingredient("IngotSteel");
var fireSalts = Ingredient("FireSalts");
var commonGem = Ingredient("SoulGemCommon");
var sharpeningWheel = Kw("CraftingSmithingSharpeningWheel");

var emberShader = Find(skyrim.EffectShaders, "EnchFireFXShader").Duplicate(Mine(0x841));
emberShader.EditorID = "HollowLanternEmberFXShader";
emberShader.FormVersion = 44;
emberShader.FillColorKey2 = Color.FromArgb(0, 255, 168, 64);
emberShader.FillColorKey3 = Color.FromArgb(0, 186, 82, 20);
emberShader.EdgeEffectColor = Color.FromArgb(0, 255, 132, 32);
emberShader.EdgeEffectFallOff = 1.4f;
emberShader.EdgeEffectAlphaPulseAmplitude = 0.55f;
emberShader.EdgeEffectAlphaPulseFrequency = 1.3f;
mod.EffectShaders.Add(emberShader);

var ember = Find(skyrim.MagicEffects, "EnchFireDamageFFContact").Duplicate(Mine(0x842));
ember.EditorID = "HollowLanternEmberDamage";
ember.Name = "Lantern Ember";
ember.Description = "Burns the target for <mag> points of fire damage.";
ember.EnchantShader.SetTo(emberShader.FormKey);
mod.MagicEffects.Add(ember);

var burn = Find(skyrim.MagicEffects, "EnchFireDamageFFContact").Duplicate(Mine(0x84D));
burn.EditorID = "HollowLanternEmberBurn";
burn.Name = "Lantern Burn";
burn.Description = "The target keeps burning for <mag> points per second for <dur> seconds. Does not stack.";
burn.EnchantShader.SetTo(emberShader.FormKey);
mod.MagicEffects.Add(burn);

var hunger = Find(skyrim.MagicEffects, "EnchSoulTrapFFContact").Duplicate(Mine(0x843));
hunger.EditorID = "HollowLanternSoulHarvest";
hunger.Name = "Lantern's Hunger";
hunger.Description = "If the target dies within <dur> seconds, the lantern draws its soul into a soul gem.";
hunger.EnchantShader.SetTo(emberShader.FormKey);
mod.MagicEffects.Add(hunger);

var dread = Find(skyrim.MagicEffects, "EnchInfluenceConfDownFFContactMed").Duplicate(Mine(0x84E));
dread.EditorID = "HollowLanternDread";
dread.Name = "Lantern's Dread";
dread.Description = "Wounded creatures and people up to level <mag> flee in terror for <dur> seconds.";
mod.MagicEffects.Add(dread);

ObjectEffect Enchantment(uint id, string editorId, uint cost, float fire, float burnRate, int dreadLevel, int dreadTime)
{
    var e = new ObjectEffect(Mine(id), release)
    {
        EditorID = editorId,
        Name = "Hollow Lantern Ember",
        CastType = CastType.FireAndForget,
        TargetType = TargetType.Touch,
        EnchantType = ObjectEffect.EnchantTypeEnum.Enchantment,
        Flags = ObjectEffect.Flag.NoAutoCalc,
        EnchantmentCost = cost,
        EnchantmentAmount = (int)cost,
    };
    e.Effects.Add(new Effect { BaseEffect = ember.ToNullableLink(), Data = new EffectData { Magnitude = fire } });
    e.Effects.Add(new Effect { BaseEffect = burn.ToNullableLink(), Data = new EffectData { Magnitude = burnRate, Duration = 4 } });
    e.Effects.Add(new Effect { BaseEffect = hunger.ToNullableLink(), Data = new EffectData { Duration = 10 } });
    var wounded = new GetActorValuePercentConditionData { RunOnType = Condition.RunOnType.Subject };
    wounded.ActorValue = ActorValue.Health;
    var fear = new Effect { BaseEffect = dread.ToNullableLink(), Data = new EffectData { Magnitude = dreadLevel, Duration = dreadTime } };
    fear.Conditions.Add(new ConditionFloat { CompareOperator = CompareOperator.LessThan, ComparisonValue = 0.35f, Data = wounded });
    e.Effects.Add(fear);
    mod.ObjectEffects.Add(e);
    return e;
}

var axeEnchant = Enchantment(0x844, "HollowLanternEmberEnch", 60, 18, 4, 35, 8);
var greatEnchant = Enchantment(0x84F, "HollowLanternEmberEnchGreat", 85, 26, 6, 45, 10);

var weapons = new[]
{
    new { Key = "waraxe", Source = "SteelWarAxe", EditorName = "WarAxe", Name = "War Axe", Id = 0x847u, StatId = 0x845u,
          RecipeId = 0x849u, TemperId = 0x84Bu, Damage = (ushort)14, Weight = 13f, Value = 650u, Crit = (ushort)7,
          Enchant = axeEnchant, Charge = (ushort)2500,
          Recipe = new[] { (steel, 3), (gold, 1), (strips, 2), (fireSalts, 1), (commonGem, 1) } },
    new { Key = "battleaxe", Source = "SteelBattleaxe", EditorName = "Battleaxe", Name = "Battleaxe", Id = 0x848u, StatId = 0x846u,
          RecipeId = 0x84Au, TemperId = 0x84Cu, Damage = (ushort)24, Weight = 22f, Value = 980u, Crit = (ushort)12,
          Enchant = greatEnchant, Charge = (ushort)3000,
          Recipe = new[] { (steel, 5), (gold, 1), (strips, 3), (fireSalts, 2), (commonGem, 1) } },
};
var disallowEnchanting = Kw("MagicDisallowEnchanting");
foreach (var w in weapons)
{
    var source = Find(skyrim.Weapons, w.Source);
    var bounds = Bounds(Item(w.Key).GetProperty("bounds"));
    var firstPerson = skyrim.Statics.First(s => s.FormKey == source.FirstPersonModel.FormKey).Duplicate(Mine(w.StatId));
    firstPerson.EditorID = $"1stPersonHollowLantern{w.EditorName}";
    firstPerson.Model = new Model { File = meshDir + w.Key + ".nif" };
    firstPerson.ObjectBounds = bounds;
    mod.Statics.Add(firstPerson);

    var weapon = source.Duplicate(Mine(w.Id));
    weapon.EditorID = $"HollowLantern{w.EditorName}";
    weapon.Name = $"Hollow Lantern {w.Name}";
    weapon.Model = new Model { File = meshDir + w.Key + ".nif" };
    weapon.ObjectBounds = bounds;
    weapon.FirstPersonModel.SetTo(firstPerson.FormKey);
    weapon.ObjectEffect.SetTo(w.Enchant.FormKey);
    weapon.EnchantmentAmount = w.Charge;
    weapon.BasicStats = new WeaponBasicStats { Damage = w.Damage, Weight = w.Weight, Value = w.Value };
    weapon.Critical!.Damage = w.Crit;
    weapon.Keywords!.Add(disallowEnchanting);
    mod.Weapons.Add(weapon);

    Recipe(w.RecipeId, $"RecipeHollowLantern{w.EditorName}", weapon.FormKey, forge, w.Recipe);
    Temper(w.TemperId, w.EditorName, weapon.FormKey, sharpeningWheel, steel);
}

mod.ModHeader.Stats.NextFormID = 0x851;

Directory.CreateDirectory(outDir);
var outPath = Path.Combine(outDir, modKey.FileName);
mod.BeginWrite.ToPath(outPath).WithNoLoadOrder().Write();
Console.WriteLine($"WROTE {outPath} ({new FileInfo(outPath).Length} bytes)");

using var back = SkyrimMod.CreateFromBinaryOverlay(outPath, release);
Console.WriteLine($"HEADER ESL={back.ModHeader.Flags.HasFlag(SkyrimModHeader.HeaderFlag.Small)} version={back.ModHeader.Stats.Version} next={back.ModHeader.Stats.NextFormID:X} records={back.ModHeader.Stats.NumRecords} masters={string.Join(",", back.ModHeader.MasterReferences.Select(m => m.Master.FileName))} author='{back.ModHeader.Author}'");
foreach (var a in back.Armors)
    Console.WriteLine($"  ARMO {a.FormKey.ID:X3} {a.EditorID}: '{a.Name?.String}' type={a.BodyTemplate?.ArmorType} slots={a.BodyTemplate?.FirstPersonFlags} AR={a.ArmorRating} W={a.Weight} V={a.Value} gnd={a.WorldModel?.Male?.Model?.File} bounds={a.ObjectBounds.First}/{a.ObjectBounds.Second} kw={string.Join(",", a.Keywords!.Select(k => k.FormKey.ID.ToString("X6")))} sound={a.PickUpSound.FormKeyNullable}");
foreach (var aa in back.ArmorAddons)
    Console.WriteLine($"  ARMA {aa.FormKey.ID:X3} {aa.EditorID}: races={aa.AdditionalRaces.Count} female={aa.WorldModel?.Female?.File} weightSlider={aa.WeightSliderEnabled.Female} firstPerson={aa.FirstPersonModel?.Female?.File}");
foreach (var w in back.Weapons)
    Console.WriteLine($"  WEAP {w.FormKey.ID:X3} {w.EditorID}: '{w.Name?.String}' anim={w.Data?.AnimationType} skill={w.Data?.Skill} dmg={w.BasicStats?.Damage} W={w.BasicStats?.Weight} V={w.BasicStats?.Value} speed={w.Data?.Speed} reach={w.Data?.Reach} ench={w.ObjectEffect.FormKeyNullable} charge={w.EnchantmentAmount} model={w.Model?.File} firstPerson={w.FirstPersonModel.FormKeyNullable} equip={w.EquipmentType.FormKeyNullable} bounds={w.ObjectBounds.First}/{w.ObjectBounds.Second} kw={string.Join(",", w.Keywords!.Select(k => k.FormKey.ID.ToString("X6")))}");
foreach (var s in back.Statics)
    Console.WriteLine($"  STAT {s.FormKey.ID:X3} {s.EditorID}: model={s.Model?.File}");
foreach (var e in back.ObjectEffects)
    Console.WriteLine($"  ENCH {e.FormKey.ID:X3} {e.EditorID}: '{e.Name?.String}' cost={e.EnchantmentCost} flags={e.Flags} effects={string.Join("+", e.Effects.Select(x => $"{x.BaseEffect.FormKey.ID:X3}(mag {x.Data?.Magnitude} dur {x.Data?.Duration} cond {x.Conditions.Count})"))}");
foreach (var m in back.MagicEffects)
    Console.WriteLine($"  MGEF {m.FormKey.ID:X3} {m.EditorID}: '{m.Name?.String}' archetype={m.Archetype.Type} enchantShader={m.EnchantShader.FormKey} script={m.VirtualMachineAdapter?.Scripts.FirstOrDefault()?.Name}");
foreach (var s in back.EffectShaders)
    Console.WriteLine($"  EFSH {s.FormKey.ID:X3} {s.EditorID}: palette={s.MembranePaletteTexture} edge={s.EdgeEffectColor}");
foreach (var l in back.Lights)
    Console.WriteLine($"  LIGH {l.FormKey.ID:X3} {l.EditorID}: '{l.Name?.String}' model={l.Model?.File} radius={l.Radius} color={l.Color} time={l.Time} flags={l.Flags} fade={l.FadeValue} W={l.Weight} V={l.Value} bounds={l.ObjectBounds.First}/{l.ObjectBounds.Second}");
foreach (var m in back.MiscItems)
    Console.WriteLine($"  MISC {m.FormKey.ID:X3} {m.EditorID}: '{m.Name?.String}' model={m.Model?.File} W={m.Weight} V={m.Value} script={string.Join(",", m.VirtualMachineAdapter!.Scripts.Select(s => $"{s.Name}({string.Join(",", s.Properties.OfType<IScriptObjectPropertyGetter>().Select(p => $"{p.Name}={p.Object.FormKey.ID:X3}"))})"))}");
foreach (var c in back.ConstructibleObjects)
    Console.WriteLine($"  COBJ {c.FormKey.ID:X3} {c.EditorID}: creates={c.CreatedObject.FormKey.ID:X3} bench={c.WorkbenchKeyword.FormKey} items={string.Join("+", c.Items!.Select(i => $"{i.Item.Count}x{i.Item.Item.FormKey.ID:X6}"))} conditions={c.Conditions.Count}");
