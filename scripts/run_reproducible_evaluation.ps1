<#
Run a complete, validated evaluation on the three annotated clips.

Usage (from the repository root, with the project Python environment activated):
  .\scripts\run_reproducible_evaluation.ps1

Non-inference validation of the log/CSV validators only:
  .\scripts\run_reproducible_evaluation.ps1 -ValidateOnly

The evaluator buffers timing rows in memory and writes its timing CSV only after
the frame loop. Interruption during inference can leave no timing CSV. A run
whose manifest remains "running" is incomplete and must not be treated as done.
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'

$ExpectedTimingHeader = 'frame,video_decode_ms,detector_wall_ms,detector_cpu_ms'
$NumberPattern = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

function Convert-LogNumber {
    param([Parameter(Mandatory = $true)][string]$Value)
    $parsed = 0.0
    $ok = [double]::TryParse(
        $Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsed
    )
    if (-not $ok -or [double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
        throw "Invalid numeric value in evaluator log: '$Value'"
    }
    return $parsed
}

function Get-RequiredLogValue {
    param(
        [Parameter(Mandatory = $true)][string]$Log,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $match = [regex]::Match($Log, $Pattern, [Text.RegularExpressions.RegexOptions]::Multiline)
    if (-not $match.Success) {
        throw "Missing or malformed $Description in evaluator log."
    }
    return $match
}

function Get-LatencyDistribution {
    param(
        [Parameter(Mandatory = $true)][string]$Log,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $escapedLabel = [regex]::Escape($Label)
    $pattern = '(?m)^' + $escapedLabel + ' \((?<count>\d+) evaluated frames, ms\):\r?\n\s*mean=(?<mean>' + $NumberPattern + ') median=(?<median>' + $NumberPattern + ') p95=(?<p95>' + $NumberPattern + ') p99=(?<p99>' + $NumberPattern + ') min=(?<min>' + $NumberPattern + ') max=(?<max>' + $NumberPattern + ')\s*$'
    $match = Get-RequiredLogValue -Log $Log -Pattern $pattern -Description "$Label summary"
    return [ordered]@{
        count = [int]$match.Groups['count'].Value
        mean_ms = Convert-LogNumber $match.Groups['mean'].Value
        median_ms = Convert-LogNumber $match.Groups['median'].Value
        p95_ms = Convert-LogNumber $match.Groups['p95'].Value
        p99_ms = Convert-LogNumber $match.Groups['p99'].Value
        min_ms = Convert-LogNumber $match.Groups['min'].Value
        max_ms = Convert-LogNumber $match.Groups['max'].Value
    }
}

function Test-ClipOutput {
    param(
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string]$TimingPath,
        [Parameter(Mandatory = $true)][int]$ExpectedFrames
    )

    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        throw "Evaluator log is missing: $LogPath"
    }
    $log = [IO.File]::ReadAllText($LogPath)
    if ($log -match '(?i)Traceback \(most recent call last\):') {
        throw 'Evaluator log contains a Python traceback.'
    }

    $resultsMarker = $log.IndexOf('EVALUATION RESULTS', [StringComparison]::Ordinal)
    if ($resultsMarker -lt 0) {
        throw 'Evaluator final EVALUATION RESULTS section is missing.'
    }
    $closingMarkers = [regex]::Matches($log, '(?m)^={50}\s*$')
    $hasClosingAfterSummary = $false
    foreach ($marker in $closingMarkers) {
        if ($marker.Index -gt $resultsMarker) {
            $hasClosingAfterSummary = $true
        }
    }
    if (-not $hasClosingAfterSummary) {
        throw 'Evaluator final closing separator is missing.'
    }
    if (-not $log.TrimEnd().EndsWith(('=' * 50), [StringComparison]::Ordinal)) {
        throw 'Evaluator log has data after, or is truncated before, the final closing separator.'
    }

    $decoded = [int](Get-RequiredLogValue $log '^Total decoded frames:\s*(\d+)\s*$' 'decoded-frame count').Groups[1].Value
    $annotated = [int](Get-RequiredLogValue $log '^Annotated frames\s*:\s*(\d+)\s*$' 'annotated-frame count').Groups[1].Value
    $evaluated = [int](Get-RequiredLogValue $log '^Frames evaluated\s*:\s*(\d+)\s*$' 'evaluated-frame count').Groups[1].Value
    $annotationRows = [int](Get-RequiredLogValue $log '^CSV annotation rows:\s*(\d+)\s*$' 'CSV annotation-row count').Groups[1].Value
    if ($decoded -ne $ExpectedFrames -or $annotated -ne $ExpectedFrames -or $evaluated -ne $ExpectedFrames -or $annotationRows -ne $ExpectedFrames) {
        throw "Frame-count mismatch: decoded=$decoded annotated=$annotated evaluated=$evaluated CSV=$annotationRows expected=$ExpectedFrames."
    }

    $tp = [int](Get-RequiredLogValue $log '^TP\s*:\s*(\d+)\s*$' 'TP').Groups[1].Value
    $fp = [int](Get-RequiredLogValue $log '^FP\s*:\s*(\d+)\s*$' 'FP').Groups[1].Value
    $fn = [int](Get-RequiredLogValue $log '^FN\s*:\s*(\d+)\s*$' 'FN').Groups[1].Value
    $precision = Convert-LogNumber (Get-RequiredLogValue $log ('^Precision\s*:\s*(' + $NumberPattern + ')\s*$') 'precision').Groups[1].Value
    $recall = Convert-LogNumber (Get-RequiredLogValue $log ('^Recall\s*:\s*(' + $NumberPattern + ')\s*$') 'recall').Groups[1].Value
    $f1 = Convert-LogNumber (Get-RequiredLogValue $log ('^F1\s*:\s*(' + $NumberPattern + ')\s*$') 'F1').Groups[1].Value
    $maxFrame = [int](Get-RequiredLogValue $log '^Detector wall-clock max frame:\s*(\d+)\s*$' 'detector maximum frame').Groups[1].Value

    if (-not (Test-Path -LiteralPath $TimingPath -PathType Leaf)) {
        throw "Timing CSV is missing: $TimingPath"
    }
    $csvLines = [IO.File]::ReadAllLines($TimingPath)
    if ($csvLines.Count -eq 0) {
        throw 'Timing CSV is empty.'
    }
    $firstLine = $csvLines[0]
    if ($firstLine -cne $ExpectedTimingHeader) {
        throw "Timing CSV header mismatch. Expected '$ExpectedTimingHeader'; got '$firstLine'."
    }
    $timingRows = @(Import-Csv -LiteralPath $TimingPath)
    if ($timingRows.Count -ne $ExpectedFrames) {
        throw "Timing CSV row count mismatch: found $($timingRows.Count), expected $ExpectedFrames."
    }
    $expectedProperties = @('frame', 'video_decode_ms', 'detector_wall_ms', 'detector_cpu_ms')
    foreach ($row in $timingRows) {
        $properties = @($row.PSObject.Properties | ForEach-Object { $_.Name })
        if (($properties -join ',') -cne ($expectedProperties -join ',')) {
            throw 'Timing CSV contains an unexpected or malformed column set.'
        }
    }
    for ($index = 0; $index -lt $ExpectedFrames; $index++) {
        $row = $timingRows[$index]
        $frame = 0
        if (-not [int]::TryParse($row.frame, [ref]$frame) -or $frame -ne ($index + 1)) {
            throw "Timing CSV frame sequence is incomplete or out of order at row $($index + 1)."
        }
        foreach ($column in @('video_decode_ms', 'detector_wall_ms', 'detector_cpu_ms')) {
            $value = 0.0
            $valid = [double]::TryParse($row.$column, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value)
            if (-not $valid -or [double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt 0) {
                throw "Timing CSV has an invalid value in '$column' at frame $($index + 1)."
            }
        }
    }

    $wall = Get-LatencyDistribution -Log $log -Label 'Detector wall-clock latency'
    $cpu = Get-LatencyDistribution -Log $log -Label 'Detector process CPU time (summed across process threads)'
    $decode = Get-LatencyDistribution -Log $log -Label 'Video decode latency (cap.read)'
    foreach ($distribution in @($wall, $cpu, $decode)) {
        if ($distribution.count -ne $ExpectedFrames) {
            throw "Latency distribution sample count $($distribution.count) does not match expected $ExpectedFrames."
        }
    }

    $outlier = Get-RequiredLogValue $log ('^  Extreme outliers \(> 2x wall p95 = (?<threshold>' + $NumberPattern + ') ms\): (?<count>\d+) / (?<total>\d+)\s*$') 'wall-time outlier summary'
    if ([int]$outlier.Groups['total'].Value -ne $ExpectedFrames) {
        throw 'Wall-time outlier denominator does not match the expected frame count.'
    }

    return [ordered]@{
        counts = [ordered]@{
            decoded = $decoded
            annotated = $annotated
            evaluated = $evaluated
            annotation_rows = $annotationRows
        }
        detection_metrics = [ordered]@{
            tp = $tp
            fp = $fp
            fn = $fn
            precision = $precision
            recall = $recall
            f1 = $f1
        }
        latency = [ordered]@{
            detector_wall = $wall
            detector_cpu = $cpu
            video_decode = $decode
            wall_outliers = [ordered]@{
                threshold_ms = Convert-LogNumber $outlier.Groups['threshold'].Value
                count = [int]$outlier.Groups['count'].Value
                total = [int]$outlier.Groups['total'].Value
            }
            detector_wall_max_frame = $maxFrame
        }
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $json = ConvertTo-Json -InputObject $Value -Depth 12
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Invoke-ValidatorSelfTest {
    $testRoot = Join-Path ([IO.Path]::GetTempPath()) ("hog_eval_runner_validation_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    try {
        $logPath = Join-Path $testRoot 'valid.log'
        $csvPath = Join-Path $testRoot 'valid.csv'
        $syntheticLog = @'
CSV annotation rows: 5
EVALUATION RESULTS
Total decoded frames: 5
Annotated frames    : 5
Frames evaluated : 5
TP                : 3
FP                : 2
FN                : 2
Precision         : 0.6000
Recall            : 0.6000
F1                : 0.6000
Detector wall-clock latency (5 evaluated frames, ms):
  mean=10.000 median=10.000 p95=12.000 p99=12.000 min=8.000 max=12.000
  Extreme outliers (> 2x wall p95 = 24.000 ms): 0 / 5
Detector process CPU time (summed across process threads) (5 evaluated frames, ms):
  mean=20.000 median=20.000 p95=22.000 p99=22.000 min=18.000 max=22.000
Video decode latency (cap.read) (5 evaluated frames, ms):
  mean=1.000 median=1.000 p95=1.200 p99=1.200 min=0.800 max=1.200
Detector wall-clock max frame: 2
==================================================
'@
        [IO.File]::WriteAllText($logPath, $syntheticLog)
        $csvLines = @($ExpectedTimingHeader)
        for ($frame = 1; $frame -le 5; $frame++) {
            $csvLines += "$frame,1.0,10.0,20.0"
        }
        [IO.File]::WriteAllLines($csvPath, $csvLines)
        $parsed = Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 5
        if ($parsed.detection_metrics.tp -ne 3 -or $parsed.counts.evaluated -ne 5 -or $parsed.latency.detector_wall.count -ne 5) {
            throw 'Synthetic validator self-test parsed unexpected values.'
        }

        $rejected = 0
        try {
            Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 6 | Out-Null
        }
        catch {
            $rejected++
        }
        if ($rejected -ne 1) {
            throw 'Synthetic validator self-test failed to reject a frame-count mismatch.'
        }

        [IO.File]::WriteAllText($logPath, $syntheticLog.Replace('EVALUATION RESULTS', 'INCOMPLETE OUTPUT'))
        $rejected = 0
        try {
            Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 5 | Out-Null
        }
        catch {
            $rejected++
        }
        if ($rejected -ne 1) {
            throw 'Synthetic validator self-test failed to reject a missing final summary.'
        }

        [IO.File]::WriteAllText($logPath, $syntheticLog.Replace('EVALUATION RESULTS', 'EVALUATION RESULTS' + "`nTraceback (most recent call last):"))
        $rejected = 0
        try {
            Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 5 | Out-Null
        }
        catch {
            $rejected++
        }
        if ($rejected -ne 1) {
            throw 'Synthetic validator self-test failed to reject a traceback.'
        }

        $csvLines[2] = '4,1.0,10.0,20.0'
        [IO.File]::WriteAllLines($csvPath, $csvLines)
        [IO.File]::WriteAllText($logPath, $syntheticLog)
        $rejected = 0
        try {
            Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 5 | Out-Null
        }
        catch {
            $rejected++
        }
        if ($rejected -ne 1) {
            throw 'Synthetic validator self-test failed to reject an incomplete frame sequence.'
        }

        $jsonPath = Join-Path $testRoot 'roundtrip.json'
        Write-JsonFile -Path $jsonPath -Value ([ordered]@{ status = 'validated'; frames = 5 })
        $roundTrip = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
        if ($roundTrip.status -ne 'validated' -or $roundTrip.frames -ne 5) {
            throw 'Synthetic validator self-test failed the JSON round trip.'
        }

        Write-Output 'VALIDATION_ONLY=PASS (valid summary/CSV and JSON round trip accepted; frame-count mismatch, incomplete summary, traceback, and frame sequence rejected; evaluator was not launched)'
    }
    finally {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}

if ($ValidateOnly) {
    Invoke-ValidatorSelfTest
    exit 0
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
$runDirectory = $null
$manifestPath = $null
$summaryPath = $null
$manifest = $null
$runFailed = $false
$failureMessage = $null
$expectedConfig = [ordered]@{
    scale = 1.05
    hit_threshold = 1.0
    win_stride = @(4, 4)
    padding = @(8, 8)
    nms_iou_threshold = 0.3
    evaluation_iou_threshold = 0.5
    sample_every = 1
    max_frames = $null
}
$clipDefinitions = @(
    [ordered]@{ name = 'crosswalk'; video = 'dataset/crosswalk.avi'; annotations = 'dataset/crosswalk.csv'; expected_frames = 378 },
    [ordered]@{ name = 'fourway'; video = 'dataset/fourway.avi'; annotations = 'dataset/fourway.csv'; expected_frames = 1281 },
    [ordered]@{ name = 'night'; video = 'dataset/night.avi'; annotations = 'dataset/night.csv'; expected_frames = 565 }
)
$runId = $null
$revision = $null

try {
    $runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    New-Item -ItemType Directory -Path (Join-Path $repoRoot 'results/reproducibility') -Force | Out-Null
    $runDirectory = Join-Path $repoRoot (Join-Path 'results/reproducibility' ("full_" + $runId))
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
    $manifestPath = Join-Path $runDirectory 'manifest.json'
    $summaryPath = Join-Path $runDirectory 'summary.json'

    $startedAt = (Get-Date).ToUniversalTime().ToString('o')
    $manifest = [ordered]@{
        run_id = $runId
        status = 'running'
        failure = $null
        repository_revision = 'unavailable'
        working_tree_dirty_at_start = $null
        working_tree_status_at_start = $null
        python = [ordered]@{
            executable = $null
            version = $null
            opencv_version = $null
        }
        started_at_utc = $startedAt
        ended_at_utc = $null
        configuration = $expectedConfig
        interruption_note = 'Timing rows are buffered in memory and written only after the evaluator frame loop. Interruption during inference may leave no timing CSV; a manifest still marked running is incomplete.'
        clips = @()
    }
    foreach ($definition in $clipDefinitions) {
        $manifest.clips += [ordered]@{
            name = $definition.name
            video = $definition.video
            annotations = $definition.annotations
            expected_frames = $definition.expected_frames
            status = 'pending'
            started_at_utc = $null
            ended_at_utc = $null
            process_exit_code = $null
            log = "$($definition.name).log"
            timings_csv = "$($definition.name)_timings.csv"
            validation_error = $null
            metrics = $null
        }
    }
    Write-JsonFile -Path $manifestPath -Value $manifest

    $revision = (& git rev-parse HEAD 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-fA-F]{40}$') {
        throw 'Could not record the full Git revision.'
    }
    $manifest.repository_revision = $revision
    $gitStatus = (& git status --porcelain 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not inspect Git working-tree status.'
    }
    $manifest.working_tree_dirty_at_start = [bool]$gitStatus
    $manifest.working_tree_status_at_start = $gitStatus
    Write-JsonFile -Path $manifestPath -Value $manifest

    $pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop
    $pythonExecutable = $pythonCommand.Source
    $pythonVersionOutput = & $pythonExecutable --version 2>&1
    $pythonVersionExitCode = $LASTEXITCODE
    if ($pythonVersionExitCode -ne 0) {
        throw "Could not query Python version (exit $pythonVersionExitCode): $($pythonVersionOutput -join ' ')"
    }
    $manifest.python.executable = $pythonExecutable
    $manifest.python.version = (($pythonVersionOutput -join ' ') -replace '^Python\s+', '').Trim()
    Write-JsonFile -Path $manifestPath -Value $manifest

    $pythonMetadataJson = & $pythonExecutable -c 'import sys, json, cv2; from src.config import DetectorConfig; from scripts.evaluate_video import IOU_THRESHOLD; c=DetectorConfig(); print(json.dumps({"executable":sys.executable,"version":sys.version.split()[0],"opencv_version":cv2.__version__,"config":{"scale":c.scale,"hit_threshold":c.hit_threshold,"win_stride":list(c.win_stride),"padding":list(c.padding),"nms_iou_threshold":c.nms_iou_threshold,"evaluation_iou_threshold":IOU_THRESHOLD}}))' 2>&1
    $pythonExitCode = $LASTEXITCODE
    if ($pythonExitCode -ne 0) {
        throw "Python/OpenCV preflight failed (exit $pythonExitCode): $($pythonMetadataJson -join ' ')"
    }
    $pythonMetadata = ($pythonMetadataJson -join "`n") | ConvertFrom-Json
    if ($pythonMetadata.config.scale -ne 1.05 -or $pythonMetadata.config.hit_threshold -ne 1.0 -or $pythonMetadata.config.nms_iou_threshold -ne 0.3 -or $pythonMetadata.config.evaluation_iou_threshold -ne 0.5 -or ($pythonMetadata.config.win_stride -join ',') -cne '4,4' -or ($pythonMetadata.config.padding -join ',') -cne '8,8') {
        throw 'Production detector or evaluation defaults differ from the requested reproducibility configuration; refusing to run.'
    }

    $manifest.python.opencv_version = $pythonMetadata.opencv_version
    Write-JsonFile -Path $manifestPath -Value $manifest

    foreach ($clip in $clipDefinitions) {
        if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $clip.video) -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $repoRoot $clip.annotations) -PathType Leaf)) {
            throw "Missing video or annotation input for $($clip.name)."
        }
    }

    Write-JsonFile -Path $manifestPath -Value $manifest

    for ($clipIndex = 0; $clipIndex -lt $clipDefinitions.Count; $clipIndex++) {
        $definition = $clipDefinitions[$clipIndex]
        $entry = $manifest.clips[$clipIndex]
        $entry.status = 'running'
        $entry.started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        Write-JsonFile -Path $manifestPath -Value $manifest

        $logPath = Join-Path $runDirectory $entry.log
        $timingPath = Join-Path $runDirectory $entry.timings_csv
        $exitCode = $null
        $clipError = $null
        try {
            & $pythonExecutable -m scripts.evaluate_video `
                --video $definition.video `
                --annotations $definition.annotations `
                --sample-every 1 `
                --timings-csv $timingPath 2>&1 |
                Tee-Object -FilePath $logPath
            $exitCode = $LASTEXITCODE
            $entry.process_exit_code = $exitCode
            if ($exitCode -ne 0) {
                throw "Evaluator process exited with code $exitCode."
            }
            $entry.metrics = Test-ClipOutput -LogPath $logPath -TimingPath $timingPath -ExpectedFrames $definition.expected_frames
            $entry.status = 'completed'
        }
        catch {
            $clipError = $_.Exception.Message
            $entry.status = 'failed'
            $entry.validation_error = $clipError
            $entry.process_exit_code = $exitCode
            $runFailed = $true
            $failureMessage = "$($definition.name): $clipError"
        }
        finally {
            $entry.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
            Write-JsonFile -Path $manifestPath -Value $manifest
        }
        if ($runFailed) {
            break
        }
    }

    if (-not $runFailed) {
        $manifest.status = 'completed'
    }
    else {
        $manifest.status = 'failed'
        for ($pendingIndex = 0; $pendingIndex -lt $manifest.clips.Count; $pendingIndex++) {
            if ($manifest.clips[$pendingIndex].status -eq 'pending') {
                $manifest.clips[$pendingIndex].status = 'not_started'
            }
        }
    }
    $manifest.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')

    $summary = [ordered]@{
        run_id = $runId
        status = $manifest.status
        repository_revision = $revision
        python = $manifest.python
        started_at_utc = $manifest.started_at_utc
        ended_at_utc = $manifest.ended_at_utc
        configuration = $expectedConfig
        interruption_note = $manifest.interruption_note
        failure = $failureMessage
        clips = @($manifest.clips)
    }
    try {
        Write-JsonFile -Path $summaryPath -Value $summary
        $null = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    }
    catch {
        $runFailed = $true
        $manifest.status = 'failed'
        $failureMessage = "summary.json creation or validation failed: $($_.Exception.Message)"
        $manifest.failure = $failureMessage
        try { Write-JsonFile -Path $manifestPath -Value $manifest } catch { }
        throw $failureMessage
    }

    if ($runFailed) {
        $manifest.status = 'failed'
        $manifest.failure = $failureMessage
        Write-JsonFile -Path $manifestPath -Value $manifest
        Write-Output "RUN_STATUS=FAILED"
        Write-Output "RUN_DIRECTORY=$runDirectory"
        Write-Error $failureMessage
    }
    else {
        Write-JsonFile -Path $manifestPath -Value $manifest
        Write-Output "RUN_STATUS=COMPLETED"
        Write-Output "RUN_DIRECTORY=$runDirectory"
    }
}
catch {
    $fatalMessage = $_.Exception.Message
    if ($manifest -and $manifestPath) {
        $manifest.status = 'failed'
        $manifest.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        $manifest.failure = $fatalMessage
        if ($summaryPath) {
            $failedSummary = [ordered]@{
                run_id = $manifest.run_id
                status = 'failed'
                repository_revision = $manifest.repository_revision
                python = $manifest.python
                started_at_utc = $manifest.started_at_utc
                ended_at_utc = $manifest.ended_at_utc
                configuration = $manifest.configuration
                interruption_note = $manifest.interruption_note
                failure = $fatalMessage
                clips = @($manifest.clips)
            }
            try {
                Write-JsonFile -Path $summaryPath -Value $failedSummary
                $null = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            }
            catch {
                $manifest.failure = "$fatalMessage; summary.json creation/validation failed: $($_.Exception.Message)"
            }
        }
        try { Write-JsonFile -Path $manifestPath -Value $manifest } catch { }
    }
    if ($runDirectory) {
        Write-Output "RUN_STATUS=FAILED"
        Write-Output "RUN_DIRECTORY=$runDirectory"
    }
    Write-Error $fatalMessage
}
finally {
    Pop-Location
}
