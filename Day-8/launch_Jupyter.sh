#!/bin/bash -l
#SBATCH -A tra26_ictpai
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time 0:30:00       # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=Jupylab
#SBATCH --output=jupyter_notebook.txt
#SBATCH --error=jupyter_notebook.err



cd $HOME


module load gcc/11.3.0

export PATH="$HOME/.venvs/fact-reasoner/bin:$SCRATCH/.local/ollama/bin:$PATH"

export MERLIN_PATH="$HOME/merlin/build/merlin"
export OLLAMA_MODELS="$SCRATCH/ollama-models"
export OLLAMA_HOST="127.0.0.1:11434"

export LD_LIBRARY_PATH="$HOME/.local/boost-1.80.0/lib:$SCRATCH/.local/ollama/lib/ollama:${LD_LIBRARY_PATH:-}"

source "$HOME/.venvs/fact-reasoner/bin/activate"



# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
portval=88$(whoami | cut -b 7-9)

#portval=8800


# print tunneling instructions jupyter-log
echo -e "
# Note: below 8888 is used to signify the port.
#       However, it may be another number if 8888 is in use.
#       Check jupyter_notebook_%j.err to find the port.

# Command to create SSH tunnel:
ssh -N -f -L $portval:${node}:$portval ${user}@login.leonardo.cineca.it
# Use a browser on your local machine to go to:
http://localhost:$portval/
"

jupyter-notebook --no-browser --ip=${node} --port=${portval}

# keep it alive
sleep 36000
