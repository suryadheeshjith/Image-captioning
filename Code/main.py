import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import argparse

from RNN import *
from coco_utils import *
from url_image import image_from_url


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',"--model",dest="model", default='rnn', help="Model. rnn or lstm. Default = rnn")
    parser.add_argument('-d',"--data",dest="train_data", type=int, default=100, help="Maximum training data. Default = 100")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    data = load_coco_data(pca_features=True)
    train_data = load_coco_data(max_train=args.train_data)
    model = Rnn(cell_type=args.model,word_to_idx=data['word_to_idx'],input_dim=data['train_features'].shape[1],hidden_dim=512,wordvec_dim=256)
    solver = Solver(model, train_data, num_epochs=50,batch_size=25,
                    optim_config={'learning_rate': 5e-3,},lr_decay=0.95,verbose=True, print_every=10)


    solver.train()

    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(train_data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            try:
                plt.imshow(image_from_url(url))
                plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
                plt.axis('off')
                plt.show()
            except:
                continue
