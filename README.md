## [Interpreting Low-level Vision Models with Causal Effect Maps](https://arxiv.org/abs/2407.19789)<br>

## Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/J-FHu/CEM.git
    cd CEM
    ```

2. Install dependent packages
    ```bash
    conda create -n CEM python=3.8 -y
    conda activate CEM
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Download Checkpoints

   Put the model checkpoints (several checkpoints are provided in [Google Drive](https://drive.google.com/drive/folders/1Ns6-LQNJSBzF6ke57vS3aeLnTQSGKxa_?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1nb86GtKsNHHyHMmpASw13g?pwd=353p) ) in “./CEM_Demo/ModelZoo/models/”.



## Quick Usage
1. Run the following command to generate Causal Effect Maps (CEMs)
    ```Shell
    export CUDA_VISIBLE_DEVICES=0
    # SR
    python demo_causal_effect_calculation.py --config demo-SR_CEM.yml
    python demo_CEM_generation.py --config demo-SR_CEM.yml
    # DR
    python demo_causal_effect_calculation.py --config demo-DR_CEM.yml
    python demo_CEM_generation.py --config demo-DR_CEM.yml
    # DN
    python demo_causal_effect_calculation.py --config demo-DN_CEM.yml
    python demo_CEM_generation.py --config demo-DN_CEM.yml
    ```
2. The hyperparameters of configure.
   ``` bash
   task: 'SR' # SR, DN, DR
   sr: 4 # 4 for SR, 1 for DN, DR
   Modellist: ['EDSR', 'SRResNet'] # Model list
   batch_size_dict: # Define batch size for each model according to your GPU memory
     EDSR: 900
     SRResNet: 800
   TextImg_path: './CEM_Demo/test_images/Test1/' # Test image path
   Imagelist: ['demo_1.png','demo_3.png'] # Test image list
   patch_path: './patch-DIV2K-8-Demo' # Intervention patch path
   prefix: 'DEMO'
   
   w: 96 # Define the top-left corner of the ROI
   h: 152 # Define the top-left corner of the ROI
   ROI_size: 32 # Define the size of the ROI
   mask_block_size: 8 # Define the size of the intervention block size
   coarse_num: 3 # Define the number of coarse-stage interventions
   fine_num: 50 # Define the number of fine-stage interventions
   load_previous: 1 # 1 or 0: Load previous intervention results or not
   tol: 0.01 # Define the tolerance of the causal effect difference
   ```


## Load Your Models
1. Put the python file of your model architecture in [./CEM_Demo/ModelZoo/NN/](./CEM_Demo/ModelZoo/NN/)

2. Edit [./CEM_Demo/ModelZoo/\_\_init__.py](./CEM_Demo/ModelZoo/__init__.py)
   - Add your **modelname** in [_NN_LIST_](./CEM_Demo/ModelZoo/__init__.py#L7)
   - Add your **modelname** and **corresponding checkpoint** in [_MODEL_LIST_](./CEM_Demo/ModelZoo/__init__.py#L81)
   - Import your model architecture in  [_get_model_](./CEM_Demo/ModelZoo/__init__.py#L370) function
   - Load model checkpoint in [_load_model_](./CEM_Demo/ModelZoo/__init__.py#L565) function

   PS: If the model is for **Denoising** (**Deraining**) task, use _get_denoise_model_ (_get_derain_model_) and _load_denoise_model_ (_get_derain_model_) instead of _get_model_ and _load_model_.
---

## BibTeX
    @article{hu2024interpreting,
      title={Interpreting Low-level Vision Models with Causal Effect Maps}, 
      author={Hu, Jinfan and Gu, Jinjin and Yu, Shiyao and Yu, Fanghua and Li, Zheyuan and You, Zhiyuan and Lu, Chaochao and Dong, Chao},
      year={2024},
      eprint={2407.19789},
      archivePrefix={arXiv},
    }
