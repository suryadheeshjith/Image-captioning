# Image-captioning
Image captioning with RNN and LSTM

#### Data
We will use the 2014 release of the Microsoft COCO dataset which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions written by workers on Amazon Mechanical Turk.

***The data preprocessing, image visualisation and encoding/decoding has been already well done by the folks at stanford university - Course CS231N. So we will be using some of the code from there.***


We have preprocessed the data and extracted features for you already. For all images we have extracted features from the fc7 layer of the VGG-16 network pretrained on ImageNet; these features are stored in the files train2014_vgg16_fc7.h5 and val2014_vgg16_fc7.h5 respectively. To cut down on processing time and memory requirements, we have reduced the dimensionality of the features from 4096 to 512; these features can be found in the files train2014_vgg16_fc7_pca.h5 and val2014_vgg16_fc7_pca.h5.

The raw images take up a lot of space (nearly 20GB) so we have not included them in the download. However all images are taken from Flickr, and URLs of the training and validation images are stored in the files train2014_urls.txt and val2014_urls.txt respectively. This allows you to download images on the fly for visualisation. Since images are downloaded on-the-fly, you must be connected to the internet to view images.

Dealing with strings is inefficient, so we will work with an encoded version of the captions. Each word is assigned an integer ID, allowing us to represent a caption by a sequence of integers. The mapping between integer IDs and words is in the file coco2014_vocab.json, and you can use the function decode_captions from the file Code/coco_utils.py to convert numpy arrays of integer IDs back into strings.
There are a couple special tokens that we add to the vocabulary. We prepend a special <START> token and append an <END> token to the beginning and end of each caption respectively. Rare words are replaced with a special <UNK> token (for "unknown"). In addition, since we want to train with minibatches containing captions of different lengths, we pad short captions with a special <NULL> token after the <END> token and don't compute loss or gradient for <NULL> tokens.

```
cd Data
./get_datasets.sh
```

If this does not work and you are interested in running the code yourself with the same data, contact me for a drive link which has all the relevant data.



#### Running the code

***Recommended to use an environment***

I am using Python version 3.7



1. Clone the repository

2. Install the requirements at Data/requirements.txt

    ```
    pip install -r requirements.txt
    ```


3. Train and sample!

```
cd Code
python3 main.py
```


RNN runs by default with 100 training samples. You can change these by passing arguments.

Optional arguments:

  MODEL : -m, --model : rnn or lstm. Default = rnn

  TRAIN_DATA : -d --data : Maximum training data. Default = 100

For example, to run LSTM with 500 training images,

```
cd Code
python3 main.py -m lstm -d 500
```


Thanks to Justin Johnson, Fei-Fei Li and Serena Yeung from the Stanford University course CS231N for the easy handling of data.

Do check out their course CS231N [here](http://cs231n.stanford.edu/2019/)
