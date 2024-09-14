Write-Host -ForegroundColor "Green" Loading voice2text

## in pwsh profile
# function bonda() {
#     (& "C:\ProgramData\Anaconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | ?{$_} | Invoke-Expression
# }

bonda
conda activate whisperx
python ./transcribe_hotkey_typer.py