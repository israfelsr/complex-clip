# ComplexCLIP

# Downloading the Benchmark

## 1. General Alignment Benchmarks
To evaluate general alignment, we utilize the following datasets:
- COCO and Flickr Retrieval: Common datasets for image captioning and image-text retrieval tasks, featuring images paired with descriptive captions.
- CIFAR-10: A dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- CIFAR-100: Similar to CIFAR-10, but with 100 classes, each containing 600 images.
- ImageNet: A large-scale dataset of images organized according to the WordNet hierarchy, primarily used for image classification.

Download Instructions:
Please follow the instructions below to download and prepare these datasets.

```
# For COCO:
# Download the Karpathy test split annotations:
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json

# Download the COCO test images (test2014):
# Ensure you are in the correct directory (e.g., your data root directory) before running this.
wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip

# For Flickr30k:
# Download the Karpathy test split annotations:
wget https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json

# For Flickr30k Images:
# The Flickr30k images require manual download due to licensing.
# You need to manually sign up and download the dataset from the following URL:
# https://forms.illinois.edu/sec/229675
# After downloading, place the image files in your designated data root directory.
# Example directory structure after manual download:
# your_data_root_dir/flickr30k/flickr30k_images/... (actual image files)
```



## 2. Compositionality Benchmarks
For evaluating compositionality, our benchmark incorporates the following dataset:

- Winoground
- SugarCrepe++ (SC++)

Winoground can be easily downloaded using the Hugging Face datasets library. Ensure you have the library installed.
