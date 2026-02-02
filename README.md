ECG-Image-FM: A Contrastive Learning Foundation Model for Electrocardiography
This repository implements a Foundation Model for ECG analysis using the principles of CLIP (Contrastive Language-Image Pre-training). By aligning 1D ECG signals with 2D ECG paper images in a joint latent space, the model learns robust, cross-modal representations that significantly enhance downstream clinical diagnosis, especially in low-data regimes.
Key Features
Modular Architecture: Signal processing, model adaptation, and datasets are fully decoupled for high maintainability.
Smart 12-Lead Preprocessing: Features a  hybrid cropping strategy that stabilizes the Y-axis (removing long leads) and precisely crops the signal area via X-axis connectivity analysis.
High Data Efficiency: Proven via Linear Probe experiments; the foundation model achieves superior performance with only a fraction of labeled data.
End-to-End Workflow: Streamlined from raw image cropping to cross-modal retrieval and multi-class AUC/ROC evaluation.

How To Use

Clone the repository and navigate to the EchoPrime directory git clone 

Download the model from the release

pip install -r requirements.txt

1. Image Preprocessing
Before training or evaluation, process raw ECG scans using the smart cropper:python scripts/ecg_img_preprocess.py --input_dir data/raw_images --output_dir data/processed_images
2. Downstream Evaluation (Linear Probe)
python src/experiments/eval_ptbxl_classification.py
