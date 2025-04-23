from sklearn.decomposition import NMF, TruncatedSVD
import numpy as np
import torch
from .build_rating_matrix import build_rating_matrix


def train_nmf_model(train_file, n_components=5, fillna='zero'):
    Z, umap, mmap = build_rating_matrix(train_file, fillna)
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(Z)
    H = model.components_
    return {'Z_approx': np.dot(W,H), 'user_map': umap, 'movie_map': mmap}


def train_svd1_model(train_file, n_components=5):
    Z, umap, mmap = build_rating_matrix(train_file)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(Z)
    Sigma = np.diag(svd.singular_values_)
    W = svd.transform(Z) / svd.singular_values_
    H = Sigma.dot(svd.components_)
    return {'Z_approx': W.dot(H), 'user_map': umap, 'movie_map': mmap}


def train_svd2_model(train_file, n_components=5, limit=10):
    Z, umap, mmap = build_rating_matrix(train_file)
    Z_approx = Z.copy()
    for _ in range(limit):
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(Z_approx)
        Sigma = np.diag(svd.singular_values_)
        W = svd.transform(Z_approx) / (svd.singular_values_ + 1e-10)
        H = Sigma.dot(svd.components_)
        Z_approx = np.where(Z!=0, Z, W.dot(H))
    return {'Z_approx': Z_approx, 'user_map': umap, 'movie_map': mmap}


def train_sgd_model(train_file, r=5, lr=0.02, n_epochs=100, device='cpu'):
    Z, umap, mmap = build_rating_matrix(train_file)
    Zt = torch.tensor(Z, dtype=torch.float, device=device)
    n, d = Zt.shape
    W = torch.randn((n, r), requires_grad=True, device=device)
    H = torch.randn((r, d), requires_grad=True, device=device)
    opt = torch.optim.SGD([W,H], lr=lr)
    for epoch in range(n_epochs):
        loss = torch.mean((Zt - W.mm(H))**2)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return {'Z_approx': W.detach().cpu().numpy().dot(H.detach().cpu().numpy()), 'user_map': umap, 'movie_map': mmap}
