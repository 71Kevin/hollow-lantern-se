<#
.SYNOPSIS
    Builds Hollow Lantern for Skyrim SE/AE and packages it with 2K, 4K and 8K textures.

.DESCRIPTION
    Reads the CBBE 3BA reference body, hands, feet, sliders and skeleton from a Mod Organizer 2 mods folder (-Mods)
    and a few vanilla meshes from the game archives, builds every piece with Blender and PyNifly (outfit meshes and
    BodySlide projects, tail physics, ground models, mask, horns, axes and lantern), paints the texture atlas and
    converts it to BC7 in three sets (8K: 8192 px, 4K: 4096 px and 2K: 2048 px, the smaller sets reduced from the
    8192 px painting), and generates "[Tinesh] Hollow Lantern.esp" with Mutagen. Each resolution is packed as a
    complete, MO2-installable archive "Hollow Lantern - CBBE 3BA (<res>) - <version>.7z" in -Packages.
    Reference data and texture sources are kept in -Work and reused.
    With -Install the chosen resolution replaces the contents of the Mod Organizer 2 mod folder (meta.ini is kept);
    MO2 must be closed.

.EXAMPLE
    .\build.ps1 -Install
#>
param(
    [string] $Version = "1.0",
    [string] $Work = "$PSScriptRoot\cache",
    [string] $Out = "$PSScriptRoot\out",
    [string] $Packages = "$PSScriptRoot\dist",
    [string] $Game = "C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition",
    [string] $Mods = "$env:LOCALAPPDATA\ModOrganizer\Skyrim Special Edition\mods",
    [string] $ModFolder = "$env:LOCALAPPDATA\ModOrganizer\Skyrim Special Edition\mods\[COSTUME] Hollow Lantern - CBBE 3BA",
    [string] $Blender = "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    [string] $SevenZip = "C:\Program Files\Vortex\resources\app.asar.unpacked\node_modules\7z-bin\win32\7z.exe",
    [ValidateSet("2K", "4K", "8K")] [string] $InstallResolution = "8K",
    [switch] $SkipMeshes,
    [switch] $SkipTextures,
    [switch] $Install
)
$ErrorActionPreference = "Continue"
$env:HL_MODS = $Mods
$tools = "$PSScriptRoot\tools\lib"
$resolutions = @("2K", "4K", "8K")
$data = Join-Path $Out "data"
$textureSets = Join-Path $Out "textures"
$report = Join-Path $Out "meshes_report.json"

$ref = Join-Path $Work "ref"
if (-not (Test-Path (Join-Path $ref "body.pkl"))) {
    python "$PSScriptRoot\tools\extract_refs.py" $ref
    if ($LASTEXITCODE) { throw "Reference extraction failed" }
}
$vanilla = Join-Path $Work "refs\vanilla"
$needed = @("meshes\weapons\torch\torch.nif", "meshes\plants\gourd01.nif", "meshes\armor\studded\male\body_go.nif",
    "meshes\weapons\steel\1stpersonsteelwaraxe.nif", "meshes\weapons\steel\1stpersonsteelbattleaxe.nif")
foreach ($bsa in @("Skyrim - Meshes0.bsa", "Skyrim - Meshes1.bsa")) {
    $missing = $needed | Where-Object { -not (Test-Path (Join-Path $vanilla $_)) }
    if ($missing) { python "$tools\bsa_get.py" "$Game\Data\$bsa" $vanilla @missing | Out-Null }
}
$missing = $needed | Where-Object { -not (Test-Path (Join-Path $vanilla $_)) }
if ($missing) { throw "Vanilla meshes not found: $($missing -join ', ')" }

if (-not $SkipMeshes) {
    foreach ($generated in @("meshes", "CalienteTools")) {
        $path = Join-Path $data $generated
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
    & $Blender --background --factory-startup --python "$PSScriptRoot\tools\build_meshes.py" -- $Work $Out |
        Select-String -Pattern "exported|written|done|Error|Traceback"
    if (-not (Test-Path $report)) { throw "Mesh build failed" }
}
if (-not $SkipTextures) {
    if (Test-Path -LiteralPath $textureSets) { Remove-Item -LiteralPath $textureSets -Recurse -Force }
    python "$PSScriptRoot\tools\make_textures.py" (Join-Path $Work "texture_manifest.pkl") $textureSets (Join-Path $Work "tex") |
        Select-String -Pattern "compressed|Error|Traceback"
    if ($LASTEXITCODE) { throw "Texture build failed" }
}
$missing = $resolutions | Where-Object { -not (Test-Path (Join-Path $textureSets $_)) }
if ($missing) { throw "Texture sets not found: $($missing -join ', ')" }

$scriptSource = Join-Path $data "Source\Scripts"
$scriptOut = Join-Path $data "Scripts"
New-Item -ItemType Directory -Force $scriptSource, $scriptOut | Out-Null
Copy-Item "$PSScriptRoot\scripts\*.psc" $scriptSource
& "$Game\Papyrus Compiler\PapyrusCompiler.exe" "$scriptSource\HollowLanternCraftScript.psc" `
    -f="$Game\Data\Source\Scripts\TESV_Papyrus_Flags.flg" -i="$scriptSource;$Game\Data\Scripts\Source;$Game\Data\Source\Scripts" `
    -o="$scriptOut" -op -q
if ($LASTEXITCODE) { throw "Papyrus compilation failed" }
python "$tools\pex_anonymize.py" "$scriptOut\HollowLanternCraftScript.pex"

dotnet run -c Release --project "$PSScriptRoot\plugin" -- "$Game\Data\Skyrim.esm" $data $report
if ($LASTEXITCODE) { throw "Plugin generation failed" }

New-Item -ItemType Directory -Force $Packages | Out-Null
foreach ($res in $resolutions) {
    $stage = Join-Path $Out "package\$res"
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    $textures = Join-Path $stage "textures\Tinesh\HollowLantern"
    New-Item -ItemType Directory -Force $textures | Out-Null
    Get-ChildItem -LiteralPath $data | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage -Recurse -Force }
    Copy-Item (Join-Path $textureSets "$res\*.dds") $textures
    $package = Join-Path $Packages "Hollow Lantern - CBBE 3BA ($res) - $Version.7z"
    if (Test-Path -LiteralPath $package) { Remove-Item -LiteralPath $package -Force }
    & $SevenZip a -t7z -mx=9 $package "$stage\*" | Out-Null
    if ($LASTEXITCODE) { throw "Packaging failed: $res" }
    "{0}  {1:N1} MB" -f $package, ((Get-Item -LiteralPath $package).Length / 1MB)
}

if ($Install) {
    if (Get-Process ModOrganizer -ErrorAction SilentlyContinue) { throw "Close Mod Organizer 2 before installing." }
    New-Item -ItemType Directory -Force -Path $ModFolder | Out-Null
    Get-ChildItem -LiteralPath $ModFolder | Where-Object { $_.Name -ne "meta.ini" } | Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath (Join-Path $Out "package\$InstallResolution") |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $ModFolder -Recurse -Force }
    Write-Host "Installed $InstallResolution to $ModFolder"
}
