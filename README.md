# Icon Matcher

Icon Matcher is a Python-based tool for comparing test icons against a reference set of icons using image similarity. This is useful for UI testing, automated comparison of assets, or categorization tasks.

## Code Review Highlights

This project was developed with careful attention to:

- **Modular Architecture:** Encapsulation of core logic inside the `IconMatcher` class.
- **Advanced Feature Extraction:** Using a pretrained ResNet50 model with global pooling and feature normalization for robust embeddings.
- **Robust Preprocessing:** Adaptive thresholding, contour cropping, and padding to standardize icon images.
- **Augmentation for References:** Rotation and flipping augmentations to improve matching under transformations.
- **Hybrid Similarity Metric:** Combining cosine similarity of features, SSIM of images, and perceptual hash distances.
- **Debugging Support:** Optional saving of preprocessed images and detailed logging for easier troubleshooting.
- **Flexible Output:** Supports CSV and JSON output formats, as well as visualizations and animated GIFs.
- **Device-aware:** Automatically uses CUDA if available, otherwise falls back to CPU.
- **Potential Improvements:** The weights used in scoring are fixed parameters; future versions could include learnable weights or saved reference embeddings for performance gains.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Daniils12334/icon-matcher.git
cd icon-matcher
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the tool from the command line with:

```bash
python icon_matcher.py --test_folder path/to/test_icons --ref_folder path/to/ref_icons --output results.csv
```

### Example:

```bash
python icon_matcher.py \
  --test_folder /home/danbar/icon-matcher/test_icons \
  --ref_folder /home/danbar/icon-matcher/ref_icons \
  --output results.csv
```

## Visualization

The project includes visualization of matching results and score distributions:

- Saved PNG images in the `visualizations/` folder show test icons alongside their top matches.
- Optionally, animated GIFs illustrate matched icon sequences.

To generate visualizations, use the `--gif` flag for GIFs and enable debug mode for preprocessed images.  


## Screenshots

> Below are examples of how visual outputs might look (when enabled):

![Discord Match](visualizations/discord.png)
![Google Match](visualizations/google.png)
![Instagram Match](visualizations/instagram.png)
![PayPal Match](visualizations/paypal.png)

<p align="center">
  <img src="visualizations/discord.png.gif" width="100"/>
  <img src="visualizations/google.png.gif" width="100"/>
  <img src="visualizations/instagram.png.gif" width="100"/>
  <img src="visualizations/paypal.png.gif" width="100"/>
</p>

## Output

The tool will generate a CSV file like this:

```
test_icon,matched_ref_icon,similarity
discord(3).png,discord.png,0.98
google.png,google(1).png,0.95
...
```

## Third-Party Notice

This project uses the ResNet-50 model architecture for feature extraction.

    If using torchvision: Model weights and architecture provided under the BSD 3-Clause License.

    Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition (2015)

## Author

Created by [Daniils12334](https://github.com/Daniils12334)
