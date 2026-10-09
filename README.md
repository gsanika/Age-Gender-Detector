# Age and Gender Detector

A lightweight Python package for estimating age and gender from face images without any C++ compilation step. It ships with bundled models, uses ONNX Runtime under the hood, and works with the standard Pillow image API.

## Features

- Face detection and age/gender estimation in a single package
- No model downloads required during normal usage
- Works with regular RGB images or cropped face images
- Includes a live webcam demo script
- Built for Python 3.11+

## Installation

```bash
pip install age-and-gender
```

Or install this version from source:

```bash
git clone https://github.com/mowshon/age-and-gender.git
cd age-and-gender
python -m pip install -e .
```

## Quick start

```python
from PIL import Image
from age_and_gender import AgeAndGender

predictor = AgeAndGender()
image = Image.open("photo.jpg").convert("RGB")

results = predictor.predict(image)
print(results)
```

Example output:

```python
[
  {
    "gender": {"value": "female", "confidence": 100},
    "age": {"value": 26, "confidence": 84},
    "face": [419, 266, 506, 352],
  }
]
```

## Using a cropped face

```python
from PIL import Image
from age_and_gender import AgeAndGender

predictor = AgeAndGender()
image = Image.open("photo.jpg").convert("RGB")

face = image.crop((100, 80, 280, 260))
print(predictor.predict_face(face))
```

## Streamlit web app

Install the optional web-app dependency from the project directory:

```bash
python -m pip install -e ".[app]"
```

Start the app:

```bash
streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). Choose
**Take a photo** to use your browser camera or **Upload an image** to select a
local image. The app draws face boxes and shows the prediction estimates. Your
browser may ask for camera permission when you choose **Take a photo**.

The Streamlit camera control captures a still photo; it does not continuously
stream video. Images are processed in memory and are not saved by the app.

## Terminal webcam demo

For continuous webcam detection in a local OpenCV window, run:

```bash
python realtime.py
```

The script opens the default camera and runs inference every few frames. Press
`Q` to quit.

## API reference

```python
from age_and_gender import AgeAndGender

predictor = AgeAndGender()
```

### Methods

- `predict(image, face_bounding_boxes=None)`
  - Returns a list of detections with `face`, `gender`, and `age`
- `predict_face(face)`
  - Runs inference on one cropped face
- `gender(face)`
  - Returns only the gender prediction
- `age(face)`
  - Returns only the age prediction
- `AgeAndGender.from_model_dir(path)`
  - Loads a custom model bundle from disk

### Inputs

- `image` or `face`: RGB `PIL.Image` or a `numpy.ndarray` shaped like `[H, W, 3]`
- `face_bounding_boxes`: optional tuple boxes in `(top, right, bottom, left)` format
- `face` in results is returned as `[left, top, right, bottom]`

## Project structure

```text
age-and-gender/
├── README.md
├── pyproject.toml
├── realtime.py
├── streamlit_app.py
├── src/
│   └── age_and_gender/
├── example/
├── tests/
└── LICENSE
```

## Credits

This project uses pretrained model assets from the dlib models ecosystem and is distributed under the MIT license. The bundled model notices are included in the package for attribution.

## License

This project is licensed under the [MIT License](LICENSE).
