import numpy as np
from tqdm import tqdm

# Visualizations
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# Classifiers
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from options import EvalOptions
from datasets import create_dataloader
from models import create_model
from utils.stats import run_classifiers, get_stats
from utils.util import load_network


def extract_features(dataloader, model):
    features = []
    labels = []

    for i, data_i in enumerate(tqdm(dataloader)):
        inputs_batch, labels_batch = data_i
        features_batch = model({'inputs': inputs_batch}, mode='eval')
        features_batch = features_batch.squeeze().cpu().data
        features.append(features_batch)
        labels.append(labels_batch)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def eval_data():
    opt = EvalOptions().parse()
    train_dataloader, test_dataloader = create_dataloader(opt)
    model = create_model(opt)

    if opt.which_epoch is not None:
        model = load_network(model, opt)

    print("Extracting train set features...")
    train_data = extract_features(train_dataloader, model)
    print("Extracting test set features...")
    test_data = extract_features(test_dataloader, model)

    classifiers = {
        # 'log_reg': LogisticRegression(solver='saga', max_iter=1000),
        # 'svm': SVC(),
        'lda': LinearDiscriminantAnalysis(),
        'rand_forest': RandomForestClassifier()
    }
    run_classifiers(classifiers, train_data, test_data)

    stats, measures = get_stats(train_data)
    for m in measures.keys():
        print('{}: '.format(m), measures[m])


if __name__ == '__main__':
    eval_data()

'''
visuals = {}

pca = PCA(n_components=2)
visuals['pca'] = pca.fit_transform(features)
print('PCA reduced features computed')

tsne_pca = PCA(n_components=opt.tsne_pca_dim)
tsne = TSNE(n_components=2, verbose=1, perplexity=opt.tsne_perplexity, n_iter=opt.tsne_iter)
features_tsne_pca = tsne_pca.fit_transform(features)
visuals['tsne'] = tsne.fit_transform(features_tsne_pca)
print('t-SNE reduced features computed')

for k in visuals.keys():
    np.save('results/{}_{}_{}.npy'.format(opt.dataset, opt.model, k), visuals[k])
'''
