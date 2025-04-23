# replicode-monai-tutorials

```bash
# Cite
https://github.com/Project-MONAI/tutorials
# Environment
vs code
# Extension
Jupyter
Python

# Set up
Select kernel: .venv(Python)

# monai_2D_classification_tutorials.ipynb
## Select monai_2D_classification_tutorials.ipynb and run each line of code
1.If you meet error'No module named sklearn', try pip install scikit-learn firstly then try again

2.If you meet error when 'pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt',
try to run 'git config --global http.sslVerify false' in Terminal, then run
'pip install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-    dev.txt' again,
then run 'git config --global http.sslVerify true' in Terminal

3.If you fail in download dataset with Python code, please try to downlaod the dataset with
'https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz' with Edge/Chrome, then put it in specific path
(for example: D:\monai_replicode\monai_datasets).
When extracting the MedNIST.tar.gz file, please run the command tar -xzvf MedNIST.tar.gz in the Terminal.
This extraction method helps avoid multi-layered nested directories (such as finding another MedNIST folder inside the extracted MedNIST folder).



