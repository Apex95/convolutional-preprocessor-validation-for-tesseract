# Validation for Image Preprocessing through Convolutions

This is the code used to evaluate the convolutional preprocessor's performance for 10,000 images in the BRNO dataset. See the article on [coding.vision](https://codingvision.net/ai/improving-tesseract-4-ocr-accuracy-through-image-preprocessing) or the more in-depth version as [published in the journal](https://www.mdpi.com/2073-8994/12/5/715). 

It uses the set of trained convolution filters avaiable in `conv-filters/cov_filters_-666.dat`. Images from the dataset are not included since they'd take too much space; they should be placed in the `data/brno/` directory and have the names mentioned in `validation.txt`.

The `plots/` directory contains the plots referenced in the article (generated when running the script).


## Running

```
python3 tester.py
```

 
