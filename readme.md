# Validation for Image Preprocessing through Convolutions

This is the code used to evaluate the convolutional preprocessor's performance for 10,000 images in the BRNO dataset. See the article on [coding.vision](https://codingvision.net/ai/improving-tesseract-4-ocr-accuracy-through-image-preprocessing) or the more in-depth version as [published in the journal](https://www.mdpi.com/2073-8994/12/5/715). 

It uses the set of trained convolution filters avaiable in `conv-filters/cov_filters_-666.dat`. Images from the dataset are not included since they'd take too much space; they should be placed in the `data/brno/` directory and have the names mentioned in `validation.txt`.

The `plots/` directory contains the plots referenced in the article (generated when running the script).


The code might be a little bit messy as I've never thought I'd actually publish it.

## Running

```
python3 tester.py
```

Expected output:
```
dan@lasher:~/work/convolutional-preprocessor$ python3 tester.py
Total chars:  569597
Avg. chars:  56.95400459954005
Avg. processed CER:  0.38477649218011123
Avg. processed WER:  0.5933691789730595
Avg. processed LCSER:  24.987501249875013
Processed CER stdev:  0.35907653087055835
Processed WER stdev:  0.4815551725494633
Processed LCSER stdev:  24.543682359532433
Avg. NON-processed CER:  0.8669550578680915
Avg. NON-processed WER:  0.9032174848878798
Avg. NON-processed LCSER:  48.83411658834117
NON-Processed CER stdev:  0.31442282213706674
NON-Processed WER stdev:  0.29750312162243775
NON-Processed LCSER stdev:  26.669697784976975
Avg. processed precision  0.7255293562720874
Avg. processed recall  0.73405452345541
Avg. NON-processed precision  0.15593248371006405
Avg. NON-processed recall  0.17280730123032162
Scores of proc vs non-proc:  3848.1496982932567 8670.417533738775
Improvement percent:  55.61748112684148
``` 
