# CLAUDE.md — Extended Project Context

## About the Developer
Khoa — Data Analytics Lead (APAC) who recently resigned from Heineken, transitioning
toward research-oriented roles. Based in Ho Chi Minh City, Vietnam.

- Master of Science in Data Science & AI Applications (AIT, December 2025)
- GPA: 3.93 (first in cohort)
- Thesis: Hybrid CNN-ViT architecture for lung segmentation
  - 96.65% Dice coefficient on Montgomery dataset
  - Cross-dataset generalization: JSRT 95.18%, Shenzhen 94.82%
  - Efficient architecture: 4.2M parameters
- 10+ years petroleum geologist before transitioning to data science
- 4+ years at Heineken: industrial IoT, predictive maintenance, Power BI
- Guest lecturer at RMIT Vietnam MBA Digital Innovation program
- Targeting PhD at University of Adelaide AIML group, February 2027

## Why This Project Exists
This is a portfolio piece that demonstrates:
1. Taking research models to production (thesis → deployed API)
2. Full MLOps lifecycle (train → register → serve → monitor → retrain)
3. Engineering maturity beyond Jupyter notebooks
4. Medical AI domain expertise aligned with PhD research direction

The narrative: "I built a 96.65% Dice model. Then I deployed it as a production API
with CI/CD, monitoring, and continuous retraining."

## Existing Thesis Assets

### Model Weights (on this WSL machine)
These are the trained model checkpoint files from the thesis:

**CNN-only model:**
- Complete checkpoint: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_bce_dice_cnn_only_complete_20250908_082124.pth
- Inference-ready: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_bce_dice_cnn_only_inference_ready_20250908_082124.pth

**ViT-only model:**
- Complete checkpoint: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_bce_dice_complete_20250919_161314.pth
- Inference-ready: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_bce_dice_inference_ready_20250919_161314.pth

**Hybrid CNN-ViT model (primary):**
- Complete checkpoint: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_complete_20250927_155222.pth
- Inference-ready: ~/AIT_LungSegmentation/saved_models_advanced/lung_segmentation_inference_ready_20250927_155222.pth

NOTE: Verify these paths exist before using. Run:
  ls -lh ~/AIT_LungSegmentation/saved_models_advanced/*.pth

### Model Architecture Source Code
The original model class definitions are in the Streamlit apps:
- CNN model class: ~/AIT_LungSegmentation/lung_seg/model_cnn.py (or similar)
- ViT model class: ~/AIT_LungSegmentation/lung_seg/model_vit.py (or similar)
- Hybrid model class: ~/AIT_LungSegmentation/lung_seg/model.py (or similar)

NOTE: Check the actual file locations. They may also be in:
  ~/AIT_LungSegmentation/a/
  ~/AIT_LungSegmentation/colab notebook/

The key classes to look for:
- CNN: look for class with ResNet18 encoder + UNet decoder
- ViT: look for class with DeiT-Tiny + progressive decoder
- Hybrid: look for SimpleHybridLungSeg or similar with cross-attention fusion

### Existing Data
- ~/AIT_LungSegmentation/data/ — training datasets
- Check for Montgomery, Shenzhen, JSRT subdirectories

### Existing Predictions
- ~/AIT_LungSegmentation/test_predictions/ — sample outputs
- ~/AIT_LungSegmentation/test_predictions_COVID/ — COVID test results
- ~/AIT_LungSegmentation/test_predictions_COVID_BCE_DICE_CNN_only/
- ~/AIT_LungSegmentation/test_predictions_COVID_BCE_DICE_Hybrid/
- ~/AIT_LungSegmentation/test_predictions_COVID_BCE_DICE_Vit_only/

## Porting Strategy
When porting model architectures from the thesis code:
1. Read the original files to understand the exact architecture
2. Clean up: add type hints, docstrings, logging (replace prints)
3. Standardize the interface: __init__, forward, predict, get_param_count
4. Do NOT modify the actual layer structure — keep architectures identical
5. The weights must load into the new code without errors

## Medical Disclaimer
All API responses and documentation must include:
"Research and educational use only. Not intended for clinical diagnosis or
medical decision-making. Always consult qualified healthcare professionals."

Only public datasets are used. No patient-identifiable information.

## License
MIT License for the project code.
Model weights are for research use only.
