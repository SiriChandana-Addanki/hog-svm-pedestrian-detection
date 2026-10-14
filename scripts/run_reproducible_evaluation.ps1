<#
Run a complete, validated evaluation on the three annotated clips.

Usage (from the repository root, with the project Python environment activated):
  .\scripts\run_reproducible_evaluation.ps1

Non-inference validation of the log/CSV validators only:
  .\scripts\run_reproducible_evaluation.ps1 -ValidateOnly

The evaluator buffers timing rows in memory and writes its timing CSV only after
the frame loop. Interruption during inference can leave no timing CSV. A run
whose manifest remains "running" is incomplete and must not be treated as done.
# A run is complete only when manifest.json and summary.json both say completed
# for the same run and completion.json exists with the matching run ID/status.
#>
[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'

$ExpectedTimingHeader = 'frame,video_decode_ms,detector_wall_ms,detector_cpu_ms'
$NumberPattern = '[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

function Format-FailureDiagnostic {
    param([Parameter(Mandatory = $true)]$ErrorRecord, [Parameter(Mandatory = $true)][string]$Stage)
    $parts = @("Stage: $Stage", "Message: $($ErrorRecord.Exception.Message)", "Exception details:`n$($ErrorRecord.Exception.ToString())")
    if ($ErrorRecord.ScriptStackTrace) { $parts += "PowerShell script stack:`n$($ErrorRecord.ScriptStackTrace)" }
    return $parts -join "`n"
}

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
        [Parameter(Mandatory = $true)][string]$Description,
        [string]$PresencePattern
    )
    $matches = [regex]::Matches($Log, $Pattern, [Text.RegularExpressions.RegexOptions]::Multiline)
    if ($PresencePattern) {
        $present = [regex]::Matches($Log, $PresencePattern, [Text.RegularExpressions.RegexOptions]::Multiline)
        if ($present.Count -ne 1) {
            if ($present.Count -gt 1) { throw "Duplicate $Description fields in evaluator log ($($present.Count) occurrences)." }
            throw "Missing $Description field in evaluator log."
        }
    }
    if ($matches.Count -ne 1) {
        if ($matches.Count -gt 1) {
            throw "Duplicate $Description fields in evaluator log ($($matches.Count) occurrences)."
        }
        throw "Missing or malformed $Description in evaluator log."
    }
    return $matches[0]
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

    $tp = [int](Get-RequiredLogValue $log '^TP\s*:\s*(\d+)\s*$' 'TP' '^TP\s*:').Groups[1].Value
    $fp = [int](Get-RequiredLogValue $log '^FP\s*:\s*(\d+)\s*$' 'FP' '^FP\s*:').Groups[1].Value
    $fn = [int](Get-RequiredLogValue $log '^FN\s*:\s*(\d+)\s*$' 'FN' '^FN\s*:').Groups[1].Value
    $precision = Convert-LogNumber (Get-RequiredLogValue $log ('^Precision\s*:\s*(' + $NumberPattern + ')\s*$') 'precision' '^Precision\s*:').Groups[1].Value
    $recall = Convert-LogNumber (Get-RequiredLogValue $log ('^Recall\s*:\s*(' + $NumberPattern + ')\s*$') 'recall' '^Recall\s*:').Groups[1].Value
    $f1 = Convert-LogNumber (Get-RequiredLogValue $log ('^F1\s*:\s*(' + $NumberPattern + ')\s*$') 'F1' '^F1\s*:').Groups[1].Value
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

function ConvertTo-CanonicalValue {
    param([AllowNull()]$Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [System.Collections.IDictionary]) {
        $canonical = [ordered]@{}
        foreach ($key in @($Value.Keys | Sort-Object)) { $canonical[$key] = ConvertTo-CanonicalValue $Value[$key] }
        return ,$canonical
    }
    if ($Value -is [string] -or $Value.GetType().IsValueType) { return $Value }
    if ($Value -is [System.Collections.IEnumerable]) {
        $items = @()
        foreach ($item in $Value) { $items += ,(ConvertTo-CanonicalValue $item) }
        return ,$items
    }
    $properties = @($Value.PSObject.Properties | Where-Object { $_.MemberType -in @('NoteProperty', 'Property') } | Sort-Object Name)
    if ($properties.Count -gt 0) {
        $canonical = [ordered]@{}
        foreach ($property in $properties) { $canonical[$property.Name] = ConvertTo-CanonicalValue $property.Value }
        return ,$canonical
    }
    return $Value
}

function Test-StructuredEqual {
    param([AllowNull()]$Left, [AllowNull()]$Right)
    $leftJson = ConvertTo-Json -InputObject (ConvertTo-CanonicalValue $Left) -Compress -Depth 20
    $rightJson = ConvertTo-Json -InputObject (ConvertTo-CanonicalValue $Right) -Compress -Depth 20
    return [string]::Equals($leftJson, $rightJson, [StringComparison]::Ordinal)
}

function Get-FieldNames {
    param([Parameter(Mandatory = $true)]$Value)
    if ($Value -is [System.Collections.IDictionary]) { return @($Value.Keys | ForEach-Object { [string]$_ } | Sort-Object) }
    return @($Value.PSObject.Properties | Where-Object { $_.MemberType -in @('NoteProperty', 'Property') } | ForEach-Object { $_.Name } | Sort-Object)
}

function Assert-FieldSet {
    param([Parameter(Mandatory = $true)]$Value, [Parameter(Mandatory = $true)][string[]]$Expected, [Parameter(Mandatory = $true)][string]$Description)
    if (-not (Test-StructuredEqual @(Get-FieldNames $Value) @($Expected | Sort-Object))) { throw "$Description has missing or unexpected fields." }
}

function Assert-SuccessArtifactPair {
    param($Manifest, $Summary, [Parameter(Mandatory = $true)]$ExpectedConfig, [Parameter(Mandatory = $true)]$ClipDefinitions, [Parameter(Mandatory = $true)][string]$ExpectedStatus, [Parameter(Mandatory = $true)][string]$ExpectedRevision)
    foreach ($field in @('status', 'run_id', 'repository_revision', 'configuration', 'failure', 'clips')) {
        if ($field -eq 'failure') {
            if (@(Get-FieldNames $Manifest) -notcontains 'failure' -or @(Get-FieldNames $Summary) -notcontains 'failure') {
                throw "Successful artifacts are missing '$field'."
            }
        }
        elseif ($null -eq $Manifest.$field -or $null -eq $Summary.$field) { throw "Successful artifacts are missing '$field'." }
    }
    if ($Manifest.status -ne $ExpectedStatus -or $Summary.status -ne $ExpectedStatus -or
        $Manifest.run_id -ne $Summary.run_id -or
        [string]::IsNullOrWhiteSpace([string]$Manifest.run_id) -or
        [string]::IsNullOrWhiteSpace([string]$Manifest.repository_revision) -or
        $Manifest.repository_revision -ne $Summary.repository_revision -or
        $Manifest.repository_revision -ne $ExpectedRevision -or
        $null -ne $Manifest.failure -or $null -ne $Summary.failure) {
        throw 'Manifest and summary identity or status did not validate consistently.'
    }
    if (-not (Test-StructuredEqual $Manifest.configuration $ExpectedConfig) -or
        -not (Test-StructuredEqual $Summary.configuration $ExpectedConfig) -or
        -not (Test-StructuredEqual $Manifest.configuration $Summary.configuration)) {
        throw 'Manifest and summary production configuration did not validate consistently.'
    }
    $expectedNames = @($ClipDefinitions | ForEach-Object { [string]$_.name } | Sort-Object)
    $manifestClips = @($Manifest.clips)
    $summaryClips = @($Summary.clips)
    foreach ($clipSet in @(@($manifestClips), @($summaryClips))) {
        $names = @($clipSet | ForEach-Object { [string]$_.name })
        if ($names.Count -ne $expectedNames.Count -or @($names | Sort-Object -Unique).Count -ne $expectedNames.Count -or
            -not (Test-StructuredEqual @($names | Sort-Object) $expectedNames)) {
            throw 'Manifest or summary has a missing, duplicate, or unexpected clip.'
        }
    }
    foreach ($definition in $ClipDefinitions) {
        $manifestClip = @($manifestClips | Where-Object { $_.name -eq $definition.name })[0]
        $summaryClip = @($summaryClips | Where-Object { $_.name -eq $definition.name })[0]
        if ($manifestClip.status -ne 'completed' -or $summaryClip.status -ne 'completed' -or
            $null -eq $manifestClip.metrics -or $null -eq $summaryClip.metrics) {
            throw "Clip '$($definition.name)' has an incomplete status or missing metrics."
        }
        foreach ($clip in @($manifestClip, $summaryClip)) {
            foreach ($metricGroup in @('counts', 'detection_metrics', 'latency')) {
                if ($null -eq $clip.metrics.$metricGroup) { throw "Clip '$($definition.name)' is missing metrics '$metricGroup'." }
            }
            Assert-FieldSet $clip.metrics.counts @('decoded', 'annotated', 'evaluated', 'annotation_rows') "Clip '$($definition.name)' counts"
            Assert-FieldSet $clip.metrics.detection_metrics @('tp', 'fp', 'fn', 'precision', 'recall', 'f1') "Clip '$($definition.name)' detection metrics"
            Assert-FieldSet $clip.metrics.latency @('detector_wall', 'detector_cpu', 'video_decode', 'wall_outliers', 'detector_wall_max_frame') "Clip '$($definition.name)' latency metrics"
            foreach ($countName in @('decoded', 'annotated', 'evaluated', 'annotation_rows')) {
                if ($null -eq $clip.metrics.counts.$countName -or $clip.metrics.counts.$countName -ne $definition.expected_frames) {
                    throw "Clip '$($definition.name)' has missing or invalid '$countName' metrics."
                }
            }
            foreach ($metricName in @('tp', 'fp', 'fn', 'precision', 'recall', 'f1')) {
                if ($null -eq $clip.metrics.detection_metrics.$metricName) { throw "Clip '$($definition.name)' is missing detection metric '$metricName'." }
            }
            foreach ($metricName in @('tp', 'fp', 'fn')) {
                $countValue = 0
                if (-not [int]::TryParse([string]$clip.metrics.detection_metrics.$metricName, [ref]$countValue) -or $countValue -lt 0) {
                    throw "Clip '$($definition.name)' has malformed detection metric '$metricName'."
                }
            }
            foreach ($metricName in @('precision', 'recall', 'f1')) {
                $rateValue = 0.0
                if (-not [double]::TryParse([string]$clip.metrics.detection_metrics.$metricName, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$rateValue) -or
                    [double]::IsNaN($rateValue) -or [double]::IsInfinity($rateValue) -or $rateValue -lt 0 -or $rateValue -gt 1) {
                    throw "Clip '$($definition.name)' has malformed detection metric '$metricName'."
                }
            }
            foreach ($latencyName in @('detector_wall', 'detector_cpu', 'video_decode', 'wall_outliers', 'detector_wall_max_frame')) {
                if ($null -eq $clip.metrics.latency.$latencyName) { throw "Clip '$($definition.name)' is missing latency metric '$latencyName'." }
            }
            foreach ($distributionName in @('detector_wall', 'detector_cpu', 'video_decode')) {
                Assert-FieldSet $clip.metrics.latency.$distributionName @('count', 'mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms') "Clip '$($definition.name)' $distributionName latency"
                if ($clip.metrics.latency.$distributionName.count -ne $definition.expected_frames) {
                    throw "Clip '$($definition.name)' has an invalid '$distributionName' sample count."
                }
                foreach ($valueName in @('mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'min_ms', 'max_ms')) {
                    $latencyValue = 0.0
                    if (-not [double]::TryParse([string]$clip.metrics.latency.$distributionName.$valueName, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$latencyValue) -or
                        [double]::IsNaN($latencyValue) -or [double]::IsInfinity($latencyValue) -or $latencyValue -lt 0) {
                        throw "Clip '$($definition.name)' has malformed '$distributionName' latency '$valueName'."
                    }
                }
                if ($clip.metrics.latency.$distributionName.min_ms -gt $clip.metrics.latency.$distributionName.median_ms -or
                    $clip.metrics.latency.$distributionName.median_ms -gt $clip.metrics.latency.$distributionName.p95_ms -or
                    $clip.metrics.latency.$distributionName.p95_ms -gt $clip.metrics.latency.$distributionName.p99_ms -or
                    $clip.metrics.latency.$distributionName.p99_ms -gt $clip.metrics.latency.$distributionName.max_ms) {
                    throw "Clip '$($definition.name)' has contradictory '$distributionName' latency quantiles."
                }
            }
            Assert-FieldSet $clip.metrics.latency.wall_outliers @('threshold_ms', 'count', 'total') "Clip '$($definition.name)' wall outliers"
            if ($clip.metrics.latency.wall_outliers.total -ne $definition.expected_frames) {
                throw "Clip '$($definition.name)' has an invalid wall-outlier denominator."
            }
            $outlierThreshold = 0.0
            $outlierCount = 0
            if (-not [double]::TryParse([string]$clip.metrics.latency.wall_outliers.threshold_ms, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$outlierThreshold) -or
                [double]::IsNaN($outlierThreshold) -or [double]::IsInfinity($outlierThreshold) -or $outlierThreshold -lt 0 -or
                -not [int]::TryParse([string]$clip.metrics.latency.wall_outliers.count, [ref]$outlierCount) -or $outlierCount -lt 0 -or $outlierCount -gt $definition.expected_frames) {
                throw "Clip '$($definition.name)' has malformed wall-outlier metrics."
            }
            $maxFrame = 0
            if (-not [int]::TryParse([string]$clip.metrics.latency.detector_wall_max_frame, [ref]$maxFrame) -or $maxFrame -lt 1 -or $maxFrame -gt $definition.expected_frames) {
                throw "Clip '$($definition.name)' has an invalid detector maximum frame."
            }
        }
        if (-not (Test-StructuredEqual $manifestClip.metrics $summaryClip.metrics)) {
            throw "Clip '$($definition.name)' metrics differ between manifest and summary."
        }
    }
}

function Assert-CompletionMarker {
    param($Marker, $Manifest, $Summary, [Parameter(Mandatory = $true)]$ExpectedConfig, [Parameter(Mandatory = $true)]$ClipDefinitions, [Parameter(Mandatory = $true)][string]$ExpectedRevision)
    Assert-SuccessArtifactPair -Manifest $Manifest -Summary $Summary -ExpectedConfig $ExpectedConfig -ClipDefinitions $ClipDefinitions -ExpectedStatus 'completed' -ExpectedRevision $ExpectedRevision
    foreach ($field in @('status', 'run_id', 'repository_revision')) {
        if ($null -eq $Marker.$field -or $Marker.$field -ne $Manifest.$field -or $Marker.$field -ne $Summary.$field) {
            throw "Completion marker '$field' does not match both successful artifacts."
        }
    }
    if ($Marker.status -ne 'completed') { throw 'Completion marker status is not completed.' }
}

function Write-FailedFinalization {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [Parameter(Mandatory = $true)][string]$ManifestPath,
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][string]$Revision,
        [Parameter(Mandatory = $true)][string]$Failure
    )
    $Manifest.status = 'failed'
    $Manifest.failure = $Failure
    $Manifest.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    $failedSummary = [ordered]@{
        run_id = $Manifest.run_id
        status = 'failed'
        repository_revision = $Revision
        python = $Manifest.python
        started_at_utc = $Manifest.started_at_utc
        ended_at_utc = $Manifest.ended_at_utc
        configuration = $Manifest.configuration
        interruption_note = $Manifest.interruption_note
        failure = $Failure
        clips = @($Manifest.clips)
    }
    Write-JsonFile -Path $ManifestPath -Value $Manifest
    Write-JsonFile -Path $SummaryPath -Value $failedSummary
    $manifestCheck = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $summaryCheck = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    if ($manifestCheck.status -ne 'failed' -or $summaryCheck.status -ne 'failed' -or
        $manifestCheck.run_id -ne $summaryCheck.run_id -or
        $manifestCheck.repository_revision -ne $summaryCheck.repository_revision -or
        $manifestCheck.failure -ne $Failure -or $summaryCheck.failure -ne $Failure) {
        throw 'Failed manifest and summary did not validate consistently.'
    }
}

function Invoke-NativeEvaluator {
    param([Parameter(Mandatory = $true)][string]$Executable, [Parameter(Mandatory = $true)][string[]]$ArgumentList, [Parameter(Mandatory = $true)][string]$LogPath)
    # Redirect streams through Start-Process so native stderr is not converted
    # into a terminating PowerShell error (including under Windows PowerShell).
    $stdoutPath = "$LogPath.stdout.tmp"
    $stderrPath = "$LogPath.stderr.tmp"
    try {
        $process = Start-Process -FilePath $Executable -ArgumentList $ArgumentList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { [IO.File]::ReadAllText($stdoutPath) } else { '' }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { [IO.File]::ReadAllText($stderrPath) } else { '' }
        # Keep warnings visible, ahead of stdout so the evaluator's final
        # summary remains the end of the log for its completeness validator.
        [IO.File]::WriteAllText($LogPath, $stderr + $stdout)
        if ($stderr) { Write-Host -NoNewline $stderr }
        if ($stdout) { Write-Host -NoNewline $stdout }
        return $process.ExitCode
    }
    finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-ValidatorSelfTest {
    $inner = [InvalidOperationException]::new('synthetic inner cause')
    $outer = [Exception]::new('synthetic reporting failure', $inner)
    $record = [Management.Automation.ErrorRecord]::new($outer, 'SyntheticFailure', [Management.Automation.ErrorCategory]::NotSpecified, $null)
    $diagnostic = Format-FailureDiagnostic -ErrorRecord $record -Stage 'synthetic validation'
    if ($diagnostic -notmatch 'synthetic validation' -or $diagnostic -notmatch 'synthetic reporting failure' -or $diagnostic -notmatch 'synthetic inner cause') {
        throw 'Synthetic exception-reporting self-test omitted the stage or inner exception.'
    }
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

        $failedManifestPath = Join-Path $testRoot 'failed-manifest.json'
        $failedSummaryPath = Join-Path $testRoot 'failed-summary.json'
        $failedManifest = [ordered]@{
            run_id = 'synthetic-run'; status = 'running'; failure = $null
            repository_revision = 'synthetic-revision'; python = @{}
            started_at_utc = 'synthetic-start'; ended_at_utc = $null
            configuration = @{}; interruption_note = 'synthetic'; clips = @()
        }
        Write-FailedFinalization -Manifest $failedManifest -ManifestPath $failedManifestPath -SummaryPath $failedSummaryPath -Revision 'synthetic-revision' -Failure 'synthetic clip failure'
        $failedManifestCheck = Get-Content -LiteralPath $failedManifestPath -Raw | ConvertFrom-Json
        $failedSummaryCheck = Get-Content -LiteralPath $failedSummaryPath -Raw | ConvertFrom-Json
        if ($failedManifestCheck.status -ne 'failed' -or $failedSummaryCheck.status -ne 'failed' -or
            $failedManifestCheck.run_id -ne $failedSummaryCheck.run_id -or
            $failedManifestCheck.repository_revision -ne $failedSummaryCheck.repository_revision -or
            $failedManifestCheck.failure -ne 'synthetic clip failure' -or $failedSummaryCheck.failure -ne 'synthetic clip failure' -or
            (Test-Path -LiteralPath (Join-Path $testRoot 'completion.json'))) {
            throw 'Synthetic failed-finalization self-test did not preserve consistent failed artifacts.'
        }

        $syntheticConfig = [ordered]@{ scale = 1.05; hit_threshold = 1.0 }
        $syntheticDefinitions = @(
            [ordered]@{ name = 'clip_a'; expected_frames = 2 },
            [ordered]@{ name = 'clip_b'; expected_frames = 3 }
        )
        $newSyntheticArtifacts = {
            param($Config, $Definitions)
            $clips = @()
            foreach ($definition in $Definitions) {
                $frames = $definition.expected_frames
                $metrics = [ordered]@{
                    counts = [ordered]@{ decoded = $frames; annotated = $frames; evaluated = $frames; annotation_rows = $frames }
                    detection_metrics = [ordered]@{ tp = 1; fp = 0; fn = 1; precision = 0.5; recall = 0.5; f1 = 0.5 }
                    latency = [ordered]@{
                        detector_wall = [ordered]@{ count = $frames; mean_ms = 1.0; median_ms = 1.0; p95_ms = 1.0; p99_ms = 1.0; min_ms = 1.0; max_ms = 1.0 }
                        detector_cpu = [ordered]@{ count = $frames; mean_ms = 1.0; median_ms = 1.0; p95_ms = 1.0; p99_ms = 1.0; min_ms = 1.0; max_ms = 1.0 }
                        video_decode = [ordered]@{ count = $frames; mean_ms = 1.0; median_ms = 1.0; p95_ms = 1.0; p99_ms = 1.0; min_ms = 1.0; max_ms = 1.0 }
                        wall_outliers = [ordered]@{ threshold_ms = 2.0; count = 0; total = $frames }
                        detector_wall_max_frame = 1
                    }
                }
                $clips += [ordered]@{ name = $definition.name; status = 'completed'; metrics = $metrics }
            }
            $manifest = [ordered]@{ status = 'finalizing'; run_id = 'synthetic'; repository_revision = 'revision-a'; failure = $null; configuration = $Config; clips = $clips }
            $summary = [ordered]@{ status = 'finalizing'; run_id = 'synthetic'; repository_revision = 'revision-a'; failure = $null; configuration = $Config; clips = @($clips) }
            $summary = ConvertFrom-Json (ConvertTo-Json -InputObject $summary -Depth 20)
            return @{ manifest = $manifest; summary = $summary }
        }
        $assertRejected = {
            param([scriptblock]$Action, [string]$CaseName)
            $didReject = $false
            try { & $Action } catch { $didReject = $true }
            if (-not $didReject) { throw "Synthetic integrity validation accepted $CaseName." }
        }
        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $pair.summary.clips[0].status = 'failed'
        & $assertRejected { Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a' } 'different clip statuses'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $pair.summary.clips[0].metrics.detection_metrics.f1 = 0.25
        & $assertRejected { Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a' } 'different clip metrics'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $pair.summary.configuration.scale = 1.1
        & $assertRejected { Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a' } 'different configurations'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $badMarker = [ordered]@{ status = 'completed'; run_id = 'synthetic'; repository_revision = 'revision-b' }
        & $assertRejected { Assert-CompletionMarker -Marker $badMarker -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedRevision 'revision-a' } 'completion marker revision mismatch'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $oneSidedMarker = [ordered]@{ status = 'completed'; run_id = 'synthetic'; repository_revision = 'revision-a' }
        $pair.summary.repository_revision = 'revision-b'
        & $assertRejected { Assert-CompletionMarker -Marker $oneSidedMarker -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedRevision 'revision-a' } 'completion marker matching only one artifact'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $pair.summary.clips = @($pair.summary.clips | Where-Object { $_.name -ne 'clip_b' })
        & $assertRejected { Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a' } 'missing required clip'

        $pair = & $newSyntheticArtifacts $syntheticConfig $syntheticDefinitions
        $pair.summary.clips[0].metrics.detection_metrics.PSObject.Properties.Remove('f1')
        & $assertRejected { Assert-SuccessArtifactPair -Manifest $pair.manifest -Summary $pair.summary -ExpectedConfig $syntheticConfig -ClipDefinitions $syntheticDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision 'revision-a' } 'missing required metric'

        $warningLogPath = Join-Path $testRoot 'warning.log'
        $nativeExitCode = Invoke-NativeEvaluator -Executable $env:ComSpec -ArgumentList @('/d', '/c', 'echo UserWarning: synthetic stderr warning 1>&2') -LogPath $warningLogPath
        if ($nativeExitCode -ne 0 -or -not ([IO.File]::ReadAllText($warningLogPath).Contains('UserWarning: synthetic stderr warning'))) {
            throw 'Synthetic native-process self-test failed to preserve a successful stderr warning.'
        }

        foreach ($case in @(
            [ordered]@{ name = 'duplicate TP'; text = $syntheticLog + "`nTP : 4`n" },
            [ordered]@{ name = 'missing precision'; text = $syntheticLog.Replace('Precision         : 0.6000', '') },
            [ordered]@{ name = 'malformed F1'; text = $syntheticLog.Replace('F1                : 0.6000', 'F1                : NaN') }
        )) {
            [IO.File]::WriteAllText($logPath, $case.text)
            $rejected = $false
            try { Test-ClipOutput -LogPath $logPath -TimingPath $csvPath -ExpectedFrames 5 | Out-Null }
            catch { $rejected = $true }
            if (-not $rejected) { throw "Synthetic validator self-test accepted $($case.name)." }
        }

        Write-Output 'VALIDATION_ONLY=PASS (failed finalization, artifact mismatch/missing-field rejection, warning capture, and evaluator-output validators passed; evaluator was not launched)'
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
$expectedPythonExecutable = Join-Path $repoRoot '.hog\Scripts\python.exe'
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

if ($PreflightOnly) {
    $preflightStage = 'repository setup'
    try {
        $repoRoot = Split-Path -Parent $PSScriptRoot
        Push-Location $repoRoot
        $preflightStage = 'project interpreter verification'
        if (-not (Test-Path -LiteralPath $expectedPythonExecutable -PathType Leaf)) {
            throw "Project interpreter is missing: $expectedPythonExecutable"
        }
        $expectedPythonExecutable = (Resolve-Path -LiteralPath $expectedPythonExecutable).Path
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $pythonMetadataJson = & $expectedPythonExecutable -c 'import sys, json, cv2; from src.config import DetectorConfig; from scripts.evaluate_video import IOU_THRESHOLD; c=DetectorConfig(); print(json.dumps(dict(executable=sys.executable,version=sys.version.split()[0],opencv_version=cv2.__version__,config=dict(scale=c.scale,hit_threshold=c.hit_threshold,win_stride=list(c.win_stride),padding=list(c.padding),nms_iou_threshold=c.nms_iou_threshold,evaluation_iou_threshold=IOU_THRESHOLD))))' 2>&1
            $pythonExitCode = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $savedErrorActionPreference }
        if ($pythonExitCode -ne 0) { throw "Project interpreter preflight failed (exit $pythonExitCode): $($pythonMetadataJson -join ' ')" }
        $pythonMetadata = ($pythonMetadataJson -join "`n") | ConvertFrom-Json
        $actualPythonExecutable = [IO.Path]::GetFullPath([string]$pythonMetadata.executable)
        if (-not [string]::Equals($actualPythonExecutable, $expectedPythonExecutable, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Interpreter identity mismatch: expected '$expectedPythonExecutable'; sys.executable reported '$actualPythonExecutable'."
        }
        if ($pythonMetadata.config.scale -ne 1.05 -or $pythonMetadata.config.hit_threshold -ne 1.0 -or $pythonMetadata.config.nms_iou_threshold -ne 0.3 -or $pythonMetadata.config.evaluation_iou_threshold -ne 0.5 -or ($pythonMetadata.config.win_stride -join ',') -cne '4,4' -or ($pythonMetadata.config.padding -join ',') -cne '8,8') {
            throw 'Production detector or evaluation defaults differ from the requested reproducibility configuration; refusing to run.'
        }
        $preflightStage = 'Git revision check'
        $revision = (& git rev-parse HEAD 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-fA-F]{40}$') { throw 'Could not record the full Git revision.' }
        $preflightStage = 'video and annotation input checks'
        foreach ($clip in $clipDefinitions) {
            $videoPath = Join-Path $repoRoot $clip.video
            $annotationPath = Join-Path $repoRoot $clip.annotations
            if (-not (Test-Path -LiteralPath $videoPath -PathType Leaf)) { throw "Missing video for $($clip.name): $videoPath" }
            if (-not (Test-Path -LiteralPath $annotationPath -PathType Leaf)) { throw "Missing annotation CSV for $($clip.name): $annotationPath" }
            $rowCount = @(Import-Csv -LiteralPath $annotationPath).Count
            Write-Output "INPUT=$($clip.name) video=$videoPath annotations=$annotationPath annotation_rows=$rowCount"
        }
        Write-Output "PYTHON_EXECUTABLE=$actualPythonExecutable"
        Write-Output "PYTHON_VERSION=$($pythonMetadata.version)"
        Write-Output "OPENCV_VERSION=$($pythonMetadata.opencv_version)"
        Write-Output "GIT_REVISION=$revision"
        Write-Output ("PRODUCTION_SETTINGS=" + (ConvertTo-Json -InputObject $pythonMetadata.config -Compress -Depth 5))
        Write-Output 'PREFLIGHT_ONLY=PASS'
        Pop-Location
        exit 0
    }
    catch {
        $preflightError = $_
        $diagnostic = Format-FailureDiagnostic -ErrorRecord $preflightError -Stage $preflightStage
        [Console]::Error.WriteLine($diagnostic)
        try { Pop-Location } catch { }
        exit 1
    }
}

$runId = $null
$revision = $null
$evaluationStage = 'interpreter preflight'

try {
    if (-not (Test-Path -LiteralPath $expectedPythonExecutable -PathType Leaf)) {
        throw "Project interpreter is missing: $expectedPythonExecutable"
    }
    $expectedPythonExecutable = (Resolve-Path -LiteralPath $expectedPythonExecutable).Path
    $pythonMetadataJson = & $expectedPythonExecutable -c 'import sys, json, cv2; from src.config import DetectorConfig; from scripts.evaluate_video import IOU_THRESHOLD; c=DetectorConfig(); print(json.dumps(dict(executable=sys.executable,version=sys.version.split()[0],opencv_version=cv2.__version__,config=dict(scale=c.scale,hit_threshold=c.hit_threshold,win_stride=list(c.win_stride),padding=list(c.padding),nms_iou_threshold=c.nms_iou_threshold,evaluation_iou_threshold=IOU_THRESHOLD))))' 2>&1
    $pythonExitCode = $LASTEXITCODE
    if ($pythonExitCode -ne 0) {
        throw "Project interpreter preflight failed (exit $pythonExitCode): $($pythonMetadataJson -join ' ')"
    }
    $pythonMetadata = ($pythonMetadataJson -join "`n") | ConvertFrom-Json
    $actualPythonExecutable = [IO.Path]::GetFullPath([string]$pythonMetadata.executable)
    if (-not [string]::Equals($actualPythonExecutable, $expectedPythonExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Interpreter identity mismatch: expected '$expectedPythonExecutable'; sys.executable reported '$actualPythonExecutable'."
    }
    if ($pythonMetadata.config.scale -ne 1.05 -or $pythonMetadata.config.hit_threshold -ne 1.0 -or $pythonMetadata.config.nms_iou_threshold -ne 0.3 -or $pythonMetadata.config.evaluation_iou_threshold -ne 0.5 -or ($pythonMetadata.config.win_stride -join ',') -cne '4,4' -or ($pythonMetadata.config.padding -join ',') -cne '8,8') {
        throw 'Production detector or evaluation defaults differ from the requested reproducibility configuration; refusing to run.'
    }

    $evaluationStage = 'run directory and manifest setup'
    $runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    New-Item -ItemType Directory -Path (Join-Path $repoRoot 'results/reproducibility') -Force | Out-Null
    $runDirectory = Join-Path $repoRoot (Join-Path 'results/reproducibility' ("full_" + $runId))
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
    $manifestPath = Join-Path $runDirectory 'manifest.json'
    $summaryPath = Join-Path $runDirectory 'summary.json'
    $completionPath = Join-Path $runDirectory 'completion.json'

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

    $manifest.python.executable = $actualPythonExecutable
    $manifest.python.version = [string]$pythonMetadata.version
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
        $evaluationStage = "evaluator clip '$($definition.name)'"
        $entry = $manifest.clips[$clipIndex]
        $entry.status = 'running'
        $entry.started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        Write-JsonFile -Path $manifestPath -Value $manifest

        $logPath = Join-Path $runDirectory $entry.log
        $timingPath = Join-Path $runDirectory $entry.timings_csv
        $exitCode = $null
        $clipError = $null
        try {
            $exitCode = Invoke-NativeEvaluator -Executable $expectedPythonExecutable -ArgumentList @(
                '-m', 'scripts.evaluate_video', '--video', $definition.video,
                '--annotations', $definition.annotations, '--sample-every', '1',
                '--timings-csv', $timingPath
            ) -LogPath $logPath
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

    $evaluationStage = 'summary finalization'
    if ($runFailed) {
        for ($pendingIndex = 0; $pendingIndex -lt $manifest.clips.Count; $pendingIndex++) {
            if ($manifest.clips[$pendingIndex].status -eq 'pending') {
                $manifest.clips[$pendingIndex].status = 'not_started'
            }
        }
        try {
            Write-FailedFinalization -Manifest $manifest -ManifestPath $manifestPath -SummaryPath $summaryPath -Revision $revision -Failure $failureMessage
        }
        catch {
            $failureMessage = "$failureMessage`nFailed-run finalization also failed: $($_.Exception.Message)"
            $manifest.failure = $failureMessage
            try { Write-JsonFile -Path $manifestPath -Value $manifest } catch { }
        }
    }
    else {
        $manifest.status = 'finalizing'
        $manifest.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        $summary = [ordered]@{
            run_id = $runId
            status = 'finalizing'
            repository_revision = $revision
            python = $manifest.python
            started_at_utc = $manifest.started_at_utc
            ended_at_utc = $manifest.ended_at_utc
            configuration = $expectedConfig
            interruption_note = $manifest.interruption_note
            failure = $null
            clips = @($manifest.clips)
        }
        try {
            Write-JsonFile -Path $manifestPath -Value $manifest
            Write-JsonFile -Path $summaryPath -Value $summary
            $manifestCheck = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            $summaryCheck = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            Assert-SuccessArtifactPair -Manifest $manifestCheck -Summary $summaryCheck -ExpectedConfig $expectedConfig -ClipDefinitions $clipDefinitions -ExpectedStatus 'finalizing' -ExpectedRevision $revision
        }
        catch {
            $runFailed = $true
            $failureMessage = "summary.json creation or validation failed: $($_.Exception.Message)"
            $manifest.status = 'failed'
            $manifest.failure = $failureMessage
            try { Write-JsonFile -Path $manifestPath -Value $manifest } catch { }
            throw $failureMessage
        }
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
        # Publish status in an order that cannot produce a completed summary while
        # the manifest still says running. completion.json is the authority.
        $manifest.status = 'completed'
        Write-JsonFile -Path $manifestPath -Value $manifest
        $summary.status = 'completed'
        Write-JsonFile -Path $summaryPath -Value $summary
        $manifestCheck = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $summaryCheck = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        Assert-SuccessArtifactPair -Manifest $manifestCheck -Summary $summaryCheck -ExpectedConfig $expectedConfig -ClipDefinitions $clipDefinitions -ExpectedStatus 'completed' -ExpectedRevision $revision
        $completion = [ordered]@{
            run_id = $runId
            status = 'completed'
            manifest = 'manifest.json'
            summary = 'summary.json'
            repository_revision = $revision
            completed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        }
        $completionTemp = Join-Path $runDirectory 'completion.json.tmp'
        Write-JsonFile -Path $completionTemp -Value $completion
        $completionCheck = Get-Content -LiteralPath $completionTemp -Raw | ConvertFrom-Json
        Assert-CompletionMarker -Marker $completionCheck -Manifest $manifestCheck -Summary $summaryCheck -ExpectedConfig $expectedConfig -ClipDefinitions $clipDefinitions -ExpectedRevision $revision
        [IO.File]::Move($completionTemp, $completionPath)
        Write-Output "RUN_STATUS=COMPLETED"
        Write-Output "RUN_DIRECTORY=$runDirectory"
    }
}
catch {
    $originalError = $_
    $diagnostic = Format-FailureDiagnostic -ErrorRecord $originalError -Stage $evaluationStage
    if ($runFailed -and $failureMessage -and $diagnostic -notlike "*$failureMessage*") {
        $diagnostic = "$diagnostic`nOriginal evaluation failure: $failureMessage"
    }
    if ($manifest -and $manifestPath) {
        $manifest.status = 'failed'
        $manifest.ended_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        $manifest.failure = $diagnostic
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
                failure = $diagnostic
                clips = @($manifest.clips)
            }
            try {
                Write-JsonFile -Path $summaryPath -Value $failedSummary
                $null = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            }
            catch {
                $artifactError = $_.Exception.Message
                $manifest.failure = "$diagnostic`nFailure artifact error: $artifactError"
            }
        }
        try { Write-JsonFile -Path $manifestPath -Value $manifest } catch { }
    }
    if ($runDirectory) {
        Write-Output "RUN_STATUS=FAILED"
        Write-Output "RUN_DIRECTORY=$runDirectory"
    }
    [Console]::Error.WriteLine($diagnostic)
    exit 1
}
finally {
    Pop-Location
}
