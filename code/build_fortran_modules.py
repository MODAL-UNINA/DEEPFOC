import subprocess
import os
import time

from pathlib import Path


def build_fortran_modules(cur_dir, orig_fortran_dir, compiled_fortran_dir):
    compiled_fortran_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(compiled_fortran_dir)

    for file in orig_fortran_dir.glob("*.f"):
        # for file in [orig_fortran_dir / "azih.f"]:
        dep_folder = orig_fortran_dir / file.stem
        dep_files = []
        if dep_folder.exists() and dep_folder.is_dir():
            dep_files = list(dep_folder.rglob("*.f"))
        print(f"Compiling {file} with dependencies {dep_files}")
        time.sleep(0.5)
        subprocess.run(
            ["python", "-m", "numpy.f2py", "-c", file, *dep_files, "-m", file.stem],
            check=True,
        )

    os.chdir(cur_dir)


if __name__ == "__main__":
    cur_dir = Path(__file__).resolve().parent
    assert cur_dir.name == "code"
    root_dir = cur_dir.parent
    orig_fortran_dir = cur_dir / "gridsearchfc"
    compiled_fortran_dir = root_dir / "code" / "utils" / "fsubroutines"
    build_fortran_modules(cur_dir, orig_fortran_dir, compiled_fortran_dir)
    print("Fortran modules built successfully")
