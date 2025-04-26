#Requires -Version 5.0
<#
.SYNOPSIS
Adds all changes, commits with a given message, pushes to the remote repository,
and logs the commit message.

.DESCRIPTION
This script automates the common Git workflow of adding all changes, committing them
with a user-provided message, and pushing the commit to the default remote repository (origin main).
It also appends the commit message along with a timestamp to a log file named 'commits.log'
in the script's directory.

.PARAMETER CommitMessage
The commit message to use for the git commit command. This parameter is mandatory.

.EXAMPLE
.\git_commit_push.ps1 -CommitMessage "Implemented feature X"
Adds all changes, commits with the message "Implemented feature X", pushes to origin main,
and logs the message to commits.log.

.EXAMPLE
.\git_commit_push.ps1 "Fixed bug Y"
Same as above, using positional parameter binding for the commit message.
#>
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$CommitMessage
)

# Define the log file path relative to the script location
$logFile = Join-Path $PSScriptRoot "commits.log"
$timeStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logEntry = "$timeStamp - $CommitMessage"

# Log the commit message
try {
    Add-Content -Path $logFile -Value $logEntry -ErrorAction Stop
    Write-Host "Commit message logged to $logFile"
}
catch {
    Write-Error "Failed to write to log file '$logFile'. Error: $_"
    # Optionally exit if logging fails, depending on requirements
    # exit 1
}

# Execute Git commands
Write-Host "Running git add ."
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Error "git add . failed."
    exit $LASTEXITCODE
}

Write-Host "Running git commit -m '$CommitMessage'"
git commit -m "$CommitMessage"
if ($LASTEXITCODE -ne 0) {
    Write-Error "git commit failed."
    # Attempt to continue to push if commit failed due to no changes
    if ($LASTEXITCODE -eq 1) {
         Write-Warning "Commit failed, possibly no changes to commit. Attempting push anyway."
    } else {
        exit $LASTEXITCODE
    }
}

Write-Host "Running git push"
git push
if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed."
    exit $LASTEXITCODE
}

Write-Host "Git add, commit, and push sequence completed successfully."
