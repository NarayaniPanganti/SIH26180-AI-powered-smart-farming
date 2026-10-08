# Krushivedha - Leaf Condition Detection (Edge AI)

ML module of our Smart India Hackathon 2026 project (PS SIH26180: *Edge AI Powered Smart Farming Assistant*).
A downward-facing camera on the mobile bot classifies each plant's leaf so that only plants that need
treatment are sprayed (pesticide or supplement vessel).

## Approach
- **Model:** MobileNetV2 (ImageNet pre-trained, frozen) + small classification head - light enough for Raspberry Pi / Jetson.
- **Input:** 224x224 RGB leaf image.  **Output:** class + confidence.
- **Safety rule:** if confidence is below a threshold (default 0.6) the bot does *not* spray and flags the plant for the farmer.
- **Edge deployment:** model is converted to TensorFlow Lite (`convert_to_tflite.py`).

## Results
> Fill in with your own measured numbers - do not leave placeholders in the final submission.

| Item | Value |
|---|---|
| Dataset | _name + link_ |
| Classes | _list_ |
| Images (train / val) | _N / N_ |
| Validation accuracy | _xx %_ |
| TFLite model size | _x MB_ |

Training curve: `training_curve.png` - Confusion matrix: `confusion_matrix.png`

## Limitations / next steps
- Trained on a limited dataset; field photos (lighting, soil, shadows) will lower accuracy until we add real farm images.
- Next: fine-tune top MobileNetV2 layers, collect field data, link detections to GPS/RTK field-map coordinates, run on Raspberry Pi.

## Team
Krushivedha - Team ID 147681
