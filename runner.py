import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Uso: python runner.py <project_name> [graph_range]")
        sys.exit(1)

    project_name = sys.argv[1]
    graph_range = int(sys.argv[2]) if len(sys.argv) >= 3 else 80

    ROOT = Path(__file__).resolve().parent           # .../IAT
    ASSETS_DIR = ROOT / "main" / "assets"
    proj_dir = ASSETS_DIR / project_name

    if not proj_dir.is_dir():
        print(f"[ERRO] Projeto não encontrado: {proj_dir}")
        sys.exit(2)

    df_dir = proj_dir / "dataframes"
    batches = sorted([p.name for p in df_dir.glob("*.csv")]) if df_dir.is_dir() else []

    print(f"Project: {project_name}")
    print(f"Assets:  {ASSETS_DIR}")
    print(f"Batches encontrados: {len(batches)}")
    for b in batches:
        print(" -", b)

    app_py = ROOT / "main" / "app.py"
    cmd = [sys.executable, str(app_py), project_name, str(graph_range)]
    subprocess.run(cmd, cwd=str(ROOT), check=True)

if __name__ == "__main__":
    main()

