<#
.SYNOPSIS
    Builds the Hollow Lantern Body Presets for CBBE 3BA and packages them.

.DESCRIPTION
    Fits the five presets with Blender on the CBBE 3BA reference body extracted by the outfit build (-Work), using
    reference presets found in a Mod Organizer 2 mods folder (-Mods), writes
    "CalienteTools\BodySlide\SliderPresets\Hollow Lantern Presets.xml" and packs it as an MO2-installable archive
    "Hollow Lantern Body Presets - CBBE 3BA - <version>.7z" in -Packages.
    With -Install the package replaces the contents of the Mod Organizer 2 mod folder (meta.ini is kept); MO2 must be
    closed.

.EXAMPLE
    .\build.ps1 -Install
#>
param(
    [string] $Version = "1.0",
    [string] $Mods = "$env:LOCALAPPDATA\ModOrganizer\Skyrim Special Edition\mods",
    [string] $Work = "$PSScriptRoot\..\cache",
    [string] $Out = "$PSScriptRoot\out",
    [string] $Packages = "$PSScriptRoot\..\dist",
    [string] $ModFolder = "$env:LOCALAPPDATA\ModOrganizer\Skyrim Special Edition\mods\[PRESET] Hollow Lantern Body Presets - CBBE 3BA",
    [string] $Blender = "C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    [string] $SevenZip = "C:\Program Files\Vortex\resources\app.asar.unpacked\node_modules\7z-bin\win32\7z.exe",
    [switch] $Install
)
$ErrorActionPreference = "Continue"
$env:HL_MODS = $Mods
$data = Join-Path $Out "data"
$xml = Join-Path $data "CalienteTools\BodySlide\SliderPresets\Hollow Lantern Presets.xml"
$body = Join-Path $Work "ref\body.pkl"
if (-not (Test-Path -LiteralPath $body)) { throw "Reference body not found, run the outfit build first: $body" }

if (Test-Path -LiteralPath $data) { Remove-Item -LiteralPath $data -Recurse -Force }
& $Blender --background --factory-startup --python "$PSScriptRoot\tools\make_presets.py" -- $Mods $body $xml (Join-Path $Out "fit_report.txt") |
    Select-String -Pattern "rms dev|Error|Traceback"
if (-not (Test-Path -LiteralPath $xml)) { throw "Preset generation failed" }

New-Item -ItemType Directory -Force $Packages | Out-Null
$package = Join-Path $Packages "Hollow Lantern Body Presets - CBBE 3BA - $Version.7z"
if (Test-Path -LiteralPath $package) { Remove-Item -LiteralPath $package -Force }
& $SevenZip a -t7z -mx=9 $package "$data\*" | Out-Null
if ($LASTEXITCODE) { throw "Packaging failed" }
"{0}  {1:N1} KB" -f $package, ((Get-Item -LiteralPath $package).Length / 1KB)

if ($Install) {
    if (Get-Process ModOrganizer -ErrorAction SilentlyContinue) { throw "Close Mod Organizer 2 before installing." }
    New-Item -ItemType Directory -Force -Path $ModFolder | Out-Null
    Get-ChildItem -LiteralPath $ModFolder | Where-Object { $_.Name -ne "meta.ini" } | Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $data | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $ModFolder -Recurse -Force }
    Write-Host "Installed to $ModFolder"
}
