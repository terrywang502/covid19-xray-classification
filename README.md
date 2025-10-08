#  Multi-Stage Transfer Learning for Joint Classification and Segmentation of Chest X-ray Images

**Author:** Huaizeng (Terry) Wang  
**Supervisor:** Dr. Elsa Cardoso-Bihlo  
**Program:** Master of Data Science, Memorial University of Newfoundland  
**Date:** August 2025  

---

##  Overview
This project develops a deep learning-based **multi-task neural network** for the automatic classification of chest X-ray images, designed to assist in the diagnosis and screening of pulmonary diseases.  
Using the **COVID-19 Radiography Dataset** from Kaggle, the model performs **four-class classification** (COVID-19, Normal, Viral Pneumonia, and Lung Opacity) and integrates **lung segmentation** as an auxiliary task to guide attention to key anatomical regions.

---

##  Three-Stage Pipeline
### **Stage 1 – Transfer Learning Baseline**
- Built with **EfficientNetB0** pretrained on ImageNet  
- Backbone frozen; only classification head trained  
- Served as the baseline model

### **Stage 2 – Fine-Tuning**
- Unfroze deeper layers (blocks 6–7) of EfficientNetB0  
- Improved domain-specific feature extraction  
- Achieved significant accuracy gain over Stage 1

### **Stage 3 – Multi-Task Learning**
- Introduced **segmentation branch** for lung mask prediction  
- Joint optimization of classification and segmentation losses  
- Guided network focus on lung areas → improved robustness and interpretability

---

##  Dataset
- **Source:** [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)  
- **Categories:**  
  - COVID-19: 3,616  
  - Normal: 10,192  
  - Viral Pneumonia: 1,345  
  - Lung Opacity: 6,012  
- Each X-ray has a **corresponding lung mask** used for segmentation guidance.

---

##  Implementation
- **Framework:** PyTorch  
- **Backbone:** EfficientNetB0  
- **Loss Functions:**  
  - Weighted Cross-Entropy (classification)  
  - Binary Cross-Entropy (segmentation)  
  - Total Loss: `L_total = L_cls + 0.5 * L_seg`  
- **Data Augmentation:** Random rotation, horizontal flip, normalization  
- **Optimizers:** Adam + ReduceLROnPlateau + Early Stopping  

---

##  Results Summary
| Stage | Model Description | Accuracy | Macro Recall | Macro F1 | Segmentation Used |
|:------|:------------------|:---------:|:-------------:|:---------:|:-----------------:|
| 1 | Transfer Learning (Frozen Backbone) | 0.86 | 0.86 | 0.86 | ❌ |
| 2 | Fine-Tuning (Unfrozen Top Layers) | 0.93 | 0.93 | 0.93 | ❌ |
| 3 | Multi-Task (Classification + Segmentation) | **0.95** | **0.96** | **0.96** | ✅ |

> The Stage 3 model achieved the best overall results, confirming that integrating lung segmentation effectively enhances classification accuracy and generalization.

---

##  Visualization
**Example Outputs:**
- Training Curves (Loss & Accuracy)
- Confusion Matrices for all stages
- Predicted vs. Ground Truth Lung Masks

*(See `/outputs` folder for full visual results)*

---

##  Tech Stack
- Python 3.10  
- PyTorch, Torchvision, Albumentations  
- NumPy, Pandas, Matplotlib, Seaborn  
- scikit-learn  

---

##  How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/covid19-xray-multitask.git
cd covid19-xray-multitask

# Install dependencies
pip install -r requirements.txt

# Run training for each stage
python train_stage1_baseline.py
python train_stage2_finetune.py
python train_stage3_multitask.py
