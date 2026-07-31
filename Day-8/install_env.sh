#!/bin/bash -l


source $HOME/Conda_init.txt
conda deactivate

module load profile/deeplrn
module load cuda/11.8
module load gcc/11.3.0
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8  
module load llvm/13.0.1--gcc--11.3.0-cuda-11.8  
module load nccl/2.14.3-1--gcc--11.3.0-cuda-11.8
module load gsl/2.7.1--gcc--11.3.0-omp
module load fftw/3.3.10--gcc--11.3.0
module load python 


mkdir -p "$HOME/.local/bin" "$HOME/.venvs"
mkdir -p "$HOME/src"

if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
fi
export PATH="$HOME/.local/bin:$PATH"

ENV_DIR="$HOME/.venvs/fact-reasoner"
test -x "$ENV_DIR/bin/python" || uv venv --python 3.11 "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" \
  "fact_reasoner==0.8.0" "mellea==0.6.0" jupyterlab ipykernel

"$ENV_DIR/bin/python" -m ipykernel install --user \
  --name fact-reasoner \
  --display-name "Python 3.11 (FactReasoner)"



OLLAMA_HOME="$SCRATCH/.local/ollama"
OLLAMA_ARCHIVE="$SCRATCH/ollama-linux-amd64.tar.zst"

mkdir -p "$OLLAMA_HOME"
curl -fL --retry 3 https://ollama.com/download/ollama-linux-amd64.tar.zst \
  -o "$OLLAMA_ARCHIVE"

command -v zstd >/dev/null 2>&1 || module load zstd
zstd -dc "$OLLAMA_ARCHIVE" | tar -xf - -C "$OLLAMA_HOME"

export PATH="$OLLAMA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_HOME/lib/ollama:${LD_LIBRARY_PATH:-}"
ollama --version


#INSTALL BOOST LIBRARIES



cd "$HOME/src"

curl -fL \
  -o boost_1_80_0.tar.gz \
  https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz

tar -xzf boost_1_80_0.tar.gz
cd boost_1_80_0

./bootstrap.sh \
  --prefix="$HOME/.local/boost-1.80.0" \
  --with-libraries=atomic,chrono,date_time,program_options,system,thread

./b2 \
  -j2 \
  variant=release \
  link=shared \
  threading=multi \
  install

#  Download and patch Merlin

cd "$HOME"

if [ ! -d "$HOME/merlin/.git" ]; then
  git clone https://github.com/radum2275/merlin "$HOME/merlin"
fi

cd "$HOME/merlin"

sed -i.bak \
  '/# Optionally specify Boost root if installed via Homebrew/,/find_package(Boost REQUIRED COMPONENTS program_options thread)/d' \
  CMakeLists.txt

grep -nE 'find_package\(Boost|homebrew' CMakeLists.txt



# Build Merlin

module unload boost/1.80.0--openmpi--4.1.4--gcc--11.3.0 \
  2>/dev/null || true

module load gcc/11.3.0

export LOCAL_BOOST_DIR="$HOME/.local/boost-1.80.0"
export LD_LIBRARY_PATH="$LOCAL_BOOST_DIR/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$HOME/merlin/build"
cd "$HOME/merlin/build"

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
  -DBoost_NO_BOOST_CMAKE=ON \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBoost_INCLUDE_DIR="$LOCAL_BOOST_DIR/include" \
  -DBoost_LIBRARY_DIR_RELEASE="$LOCAL_BOOST_DIR/lib" \
  -DBoost_LIBRARY_DIR_DEBUG="$LOCAL_BOOST_DIR/lib" \
  -DBOOST_ROOT="$LOCAL_BOOST_DIR" \
  -DBOOST_INCLUDEDIR="$LOCAL_BOOST_DIR/include" \
  -DBOOST_LIBRARYDIR="$LOCAL_BOOST_DIR/lib" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMERLIN_BUILD_TESTS=OFF \


cmake --build . --parallel 8
./merlin --help


export MERLIN_PATH="$HOME/merlin/build/merlin"


"$HOME/.local/bin/uv" pip install \
  --python "$HOME/.venvs/fact-reasoner/bin/python" \
  pysqlite3-binary


source "$HOME/.venvs/fact-reasoner/bin/activate"
jupyter kernelspec list
test -d "$SCRATCH/ollama-models/manifests" && echo "Ollama model store found"