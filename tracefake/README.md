# TraceFake - AI Image Authenticity Checker 🕵️

TraceFake is a Python-based tool designed to detect Deepfakes and AI-generated images. It combines Deep Learning (EfficientNet) with Digital Forensics (ELA, EXIF) to provide a comprehensive analysis.

## 🚀 Installation

1.  **Clone the repository** (or navigate to the project folder).
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏗️ Project Structure

```
tracefake/
├── data/               # Place training data here (see below)
├── models/
│   ├── train_model.py  # Script to train the AI
│   ├── saved_model/    # Trained .h5 model will be saved here
├── forensics/          # ELA and EXIF modules
├── utils/              # Helper functions
└── app.py              # Main Streamlit Application
```

## 🧠 Training the Model

**Note**: The repository does not come with a pre-trained model due to file size and dataset licensing. You must train it yourself or place a compatible `.h5` model in `models/saved_model/tracefake_v1.h5`.

1.  **Prepare Data**:
    Structure your data folder as follows:
    ```
    tracefake/data/
    ├── train/
    │   ├── real/  (Images of real people/scenes)
    │   └── fake/  (AI generated images)
    ```
2.  **Run Training**:
    ```bash
    cd models
    python train_model.py
    ```
    This will save the best model to `saved_model/tracefake_v1.h5`.

## 🖥️ Running the App

To run the web interface:

```bash
streamlit run app.py
```

Upload an image (JPG/PNG) to see:

1.  **Prediction**: Information on whether it's Real or Fake.
2.  **Confidence Score**.
3.  **EXIF Data**: Metadata hidden in the file.
4.  **ELA**: Error Level Analysis visualization to spot retouching.

## 🧪 Verification (Forensics)

Even without a trained model, the **Forensics** tabs (EXIF and ELA) will fully function.

- **ELA**: rapid color changes or high brightness in specific areas often indicate manipulation.
- **EXIF**: Look for "Software" tags (e.g., "Adobe Photoshop") or missing camera data.
