#!/bin/bash -l
#SBATCH -A tra26_ictpai
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time 1:15:00       # format: HH:MM:SS
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=10000
#SBATCH --job-name=Jupylab
#SBATCH --output=jupyter_notebook.txt
#SBATCH --error=jupyter_notebook.err

DATA_DIR=/leonardo_scratch/large/userexternal/sdigioia/local_tutorial_walrus/

cd $DATA_DIR

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



#ONLY for walrus

source "/leonardo_scratch/large/userexternal/sdigioia/venvs/walrus-cu118/bin/activate"



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
