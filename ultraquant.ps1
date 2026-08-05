<#
.SYNOPSIS
    UltraQuant launcher for PowerShell.

.DESCRIPTION
    Opens the desktop app, or runs any other entry point with -Mode.
    Double-clicking is easier with UltraQuant.bat; this script is for people who
    live in a terminal and want the other entry points without remembering the
    module paths.

.PARAMETER Mode
    gui        the desktop application (default)
    chat       the terminal Chat/Interpreter
    forge      build a model from scratch
    bench      benchmark the execution tiers
    demo       the hybrid quantum/classical pattern demo
    paging     the on-demand shard paging demonstration
    test       the full test suite

.PARAMETER Home
    Session folder for the gui and chat modes. Defaults to .\uq_home.

.EXAMPLE
    .\ultraquant.ps1
.EXAMPLE
    .\ultraquant.ps1 -Mode forge -Rest '--synthetic','128','--compare'
#>
[CmdletBinding()]
param(
    [ValidateSet('gui', 'chat', 'forge', 'bench', 'demo', 'paging', 'test')]
    [string]$Mode = 'gui',
    [string]$Home = './uq_home',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$python = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if (-not $python) { $python = (Get-Command py -ErrorAction SilentlyContinue)?.Source }
if (-not $python) {
    Write-Host ''
    Write-Host '  Python was not found. Install 3.11+ from https://www.python.org/downloads/'
    Write-Host '  and tick "Add python.exe to PATH".'
    Write-Host ''
    exit 1
}

switch ($Mode) {
    'gui' {
        & $python -c 'import tkinter' 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host '  This Python has no Tkinter; falling back to the terminal interface.'
            & $python -m ultraquant.interpreter.chat --root $Home @Rest
            exit $LASTEXITCODE
        }
        & $python -m ultraquant.gui $Home @Rest
    }
    'chat'   { & $python -m ultraquant.interpreter.chat --root $Home @Rest }
    'forge'  { & $python -m ultraquant.forge.build @Rest }
    'bench'  { & $python -m ultraquant.bench @Rest }
    'demo'   { & $python -m ultraquant.demo @Rest }
    'paging' { & $python -m ultraquant.shards.scale_demo @Rest }
    'test'   { & $python -m unittest discover -s tests @Rest }
}
exit $LASTEXITCODE
