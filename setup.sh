module load python3/intel/3.6.3 
virtualenv --system-site-packages py3.6.3
source py3.6.3/bin/activate
pip3 install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip3 install torchtext
