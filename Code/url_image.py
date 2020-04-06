from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile
import tensorflow as tf

def image_from_url(url):

    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = tf.io.read_file(fname)
        img = tf.image.decode_image(img, channels=3)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
