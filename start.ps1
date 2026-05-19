Write-Host -ForegroundColor "Green" Loading voice2text

## in pwsh profile
# function bonda() {
#     (& "C:\ProgramData\Anaconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | ?{$_} | Invoke-Expression
# }

# bonda
# conda activate whisperx
# python ./transcribe_hotkey_typer.py

$Env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="true"
if (-not $Env:TOGGLE_INACTIVITY_TIMEOUT_SECONDS) {
    $Env:TOGGLE_INACTIVITY_TIMEOUT_SECONDS="100"
}

# Sound values may be Windows sound aliases, such as SystemAsterisk,
# SystemExclamation, SystemHand, SystemQuestion, or SystemExit, or paths to
# audio files. Set a value to an empty string to disable that event sound.
if ($null -eq $Env:SOUND_TOGGLE_ON) {
    $Env:SOUND_TOGGLE_ON="SystemAsterisk"
}
if ($null -eq $Env:SOUND_TOGGLE_OFF) {
    $Env:SOUND_TOGGLE_OFF="SystemHand"
}
if ($null -eq $Env:SOUND_TOGGLE_STILL_LISTENING) {
    $Env:SOUND_TOGGLE_STILL_LISTENING="SystemQuestion"
}
if ($null -eq $Env:SOUND_PUSH_TO_TALK_PRESS) {
    $Env:SOUND_PUSH_TO_TALK_PRESS="SystemDefault"
}
if ($null -eq $Env:SOUND_PUSH_TO_TALK_RELEASE) {
    $Env:SOUND_PUSH_TO_TALK_RELEASE="SystemExit"
}
uv run .\transcribe_hotkey_typer.py
