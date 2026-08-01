# Do Flat Minima Improve Sparse Novel View Synthesis? (ECCV 2026)


[![arXiv](https://img.shields.io/badge/arXiv-2505.01235-006600)](https://arxiv.org/abs/2511.17918) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://bbangsik13.github.io/FASR/)

[Youngsik Yun](https://bbangsik13.github.io/), [Dongjun Gu](https://github.com/dongjunKu/),
[Youngjung Uh](https://sites.google.com/yonsei.ac.kr/vi-lab/members/professor?authuser=0)<sup>†</sup>

Yonsei University &emsp;<sup>†</sup> Corresponding Author

---


Official repository for the paper "Do Flat Minima Improve Sparse Novel View Synthesis?".


## Environmental Setup
```bash
git clone https://github.com/bbangsik13/SASR.git
cd SASR
conda create -n sasr python=3.10
conda activate sasr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
pip install submodules/diff-gaussian-rasterization-w-depth/ --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation
pip install kagglehub
```
The code has been tested on RTX 2080 Ti and RTX A5000 with pytorch=2.0.0+cu118.



## Data Preparation
```bash
bash scripts/prepare_datasets.sh
```
<b>Note</b>: While testing the code, we observed differences between our reproduced results and the reported quantitative results due to differences in the COLMAP MVS reconstruction. Therefore, we provide the MVS output generated during our testing as the default dataset. 
<br>
You may regenerate the MVS results if needed. In that case, the `rho` values, except for the fixed `rho_xyz`, may require tuning. However, as discussed in the paper, applying our method only to the Gaussian means is a simplified yet highly effective approach, and we found it to be robust during reproduction.

## Run
The main experiments are run using `nohup`, and the logs are saved in the `logs` folder and results saved in the `output` folder. Please modify as needed.

```bash
bash scripts/run_llff.sh
bash scripts/run_mipnerf360.sh
```


## Bibtex
```
@misc{yun2025sasr,
          title={Do Flat Minima Improve Sparse Novel View Synthesis?}, 
          author={Youngsik Yun and Dongjun Gu and Youngjung Uh},
          year={2025},
          eprint={2511.17918},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2511.17918}, 
}
```
