# Semantic Segmentation of Solar Granulation from IMaX to DKIST

[Paper](https://journals.flvc.org/FLAIRS/article/view/138987)

<img width="1401" alt="Screenshot 2025-05-25 at 11 51 43 PM" src="https://github.com/user-attachments/assets/fc1d4024-6665-44de-8610-223d29dc42fc" />


This repository contains the codebase and results for our paper presented at the 38th Florida Artificial Intelligence Research Society (FLAIRS) Conference. We explored semantic segmentation of solar granulation—small convection-driven patterns on the Sun’s surface—using U-Net, U-Net++, and BT-UNet architectures.

We trained on annotated IMaX/SUNRISE images and applied the best models to high-resolution DKIST imagery, producing the first set of pre-annotations for this dataset.

## Highlights

- Multi-class pixel-wise segmentation of solar granulation patterns
- Evaluation of U-Net, U-Net++, and BT-UNet models across multiple architectural complexities
- Use of mIoU and Lovász-Softmax loss functions for handling class imbalance
- Pre-training using Barlow Twins redundancy reduction
- Domain transfer from IMaX to DKIST with histogram and spatial alignment

## Directory Structure

```
├── configs/                 # Model configuration files (U-Net, U-Net++, BT-UNet)
├── data/                   # IMaX dataset (original + augmented)
├── models/                 # U-Net and U-Net++ implementations
├── results/                # Saved metrics, confusion matrices, and model outputs
├── train.py                # Training entry point
├── tester.py               # Validation and scoring
├── pretrain.py             # Pre-training (Barlow Twins)
├── transform_dkist.py      # DKIST preprocessing
├── dkist_pred.py           # DKIST prediction pipeline
├── imax_pred.py            # IMaX inference script
├── losses.py, metrics.py   # Custom loss functions and metrics
└── utils.py                # Helper functions
```
## Getting Started

1. Clone the repo:
```
   git clone https://github.com/yourusername/imax-to-dkist.git
   cd imax-to-dkist
```
3. Install requirements:
   `pip install -r requirements.txt`

4. Train a model:
   `python train.py --config configs/unet/miou/64_512_miou.json`

5. Predict on DKIST frames:
   `python dkist_pred.py --model-path results/unet_64_512_miou/.../best.pth`

## Citation

If you use this code or dataset, please cite our work:

@article{Mansouri_Angryk_Reardon_2025,
  title     = {Exploring Solar Granulation: from IMaX/SUNRISE to DKIST},
  author    = {Mansouri, Reza and Angryk, Rafal and Reardon, Kevin},
  journal   = {The International FLAIRS Conference Proceedings},
  volume    = {38},
  number    = {1},
  year      = {2025},
  month     = {May},
  url       = {https://journals.flvc.org/FLAIRS/article/view/138987},
  DOI       = {10.32473/flairs.38.1.138987}
}


## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
