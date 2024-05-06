# Machine_To_Structure_TL

This repo reproduces the results of the MSSP article "Transferring Damage Detection Knowledge Across Rotating Machines and Framed Structures: Harnessing Domain Adaptation and Contrastive Learning", studying the possibility of leveraging fault detection knowledge from rotating machinery to frame structures. Two Jupyter notebooks are offered here, one ("CL-Training.ipynb") to train your models on the rotating machinery (RM) dataset that can be stored for later accuracy estimation aside from the pre-trained models we used to write that article. The step-by-step "SDD-Inference" notebook produces all the outcomes of the paper, and you can find the coding details in the "Ytils.py" file. Please do not hesitate to contact me (soleimanisam92@g.ucla.edu) with any questions you may have. Thank you.

## How to use the repo:
* 0- Intsall requierments
* 1- Cloning the repo in your system:
```bash 
cd /path/to/directory
git clone https://github.com/Hesam-92-19/Machine_To_Structure_TL.git
```
* 2- Run either CL-Training or SDD-Inference notebooks for training and SDD accuracy reports, respectively.




