import os,subprocess,tarfile,shutil,tempfile
import urllib.request
import ctypes.util
from typing import List

class Build:
  BLAS_VERSION = "0.3.29"
  @staticmethod
  def check_blas(add_names:List[str]=[]):
    blas_names = ["blas","openblas","mkl_rt"]
    if add_names: blas_names = blas_names + add_names
    for name in blas_names:
      blas = ctypes.util.find_library(name)
      if blas: 
        print(f"Pustaka BLAS ditemukan: {name} in {blas}")
        return blas
    print("Pustaka BLAS tidak ditemukan !")
    return False
  
  @staticmethod
  def install_openblas():
    openblas_version = "0.3.29"
    # url : https://github.com/OpenMathLib/OpenBLAS/archive/v0.3.29.tar.gz
    openblas_url = f"https://github.com/OpenMathLib/OpenBLAS/archive/v{openblas_version}.tar.gz"
    lib_dir = os.path.join(os.getcwd(),"blas")
    if os.path.exists(lib_dir): shutil.rmtree(lib_dir)
    os.makedirs(lib_dir,exist_ok=True)

    print(f"Mengunduh OpenBLAS-{Build.BLAS_VERSION} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
      openblas_tar = os.path.join(tmpdir,"openblas.tar.gz")
      openblas_dir = os.path.join(tmpdir,f"OpenBLAS-{Build.BLAS_VERSION}")
      urllib.request.urlretrieve(openblas_url,openblas_tar)
      print("downloaded successfully !")

      with tarfile.open(openblas_tar,"r:gz") as tar: tar.extractall(path=tmpdir)
      print("Extracted Successfully")
      os.chdir(openblas_dir)

      # Compile
      make_cmd = ["make", "-j4", "NO_STATIC=0", "NO_SHARED=0"]
      subprocess.run(make_cmd)

      # Install
      install_cmd = ["make",f"PREFIX={lib_dir}","install"]
      subprocess.run(install_cmd)

    print("✅ Build & install selesai di:", lib_dir)

  @staticmethod
  def setup():
    if os.environ.get("BLAS") == "1": Build.install_openblas()
    if Build.check_blas(): return
    else: Build.install_openblas()

if __name__ == "__main__":
  Build.setup()



