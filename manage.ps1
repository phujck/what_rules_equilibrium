param(
    [string]$Command
)

$PythonStr = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonStr = "py"
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $PythonStr = "python3"
}

if ($Command -eq "sim") {
    & $PythonStr simulations/src/run_what_rules_suite.py --regime all --profile full --seed 42
    & $PythonStr simulations/src/plot_what_rules_suite.py
    & $PythonStr simulations/src/validate_what_rules_claims.py
}
elseif ($Command -eq "sim-ma") {
    & $PythonStr simulations/src/run_what_rules_suite.py --regime cg --profile full --seed 42
    & $PythonStr simulations/src/plot_what_rules_suite.py
}
elseif ($Command -eq "sim-qc") {
    & $PythonStr simulations/src/run_what_rules_suite.py --regime ncg --profile full --seed 42
    & $PythonStr simulations/src/plot_what_rules_suite.py
}
elseif ($Command -eq "paper") {
    Push-Location manuscript/tex
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
    Pop-Location
}
else {
    Write-Host "Usage: ./manage.ps1 [sim|sim-ma|sim-qc|paper]"
}
