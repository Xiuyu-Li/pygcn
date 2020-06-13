import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from networkx.readwrite import json_graph
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, diags, identity
import scipy.sparse as sp
import process
import json
import sys
import pickle as pkl


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def json2mtx(dataset):
    G_data    = json.load(open("/home/xl289/CS6241_proj/dataset/{}/{}-G.json".format(dataset, dataset)))
    G         = json_graph.node_link_graph(G_data)
    laplacian = laplacian_matrix(G)
    file = open("/home/xl289/CS6241_proj/dataset/{}/{}.mtx".format(dataset, dataset), "wb")
    mmwrite("/home/xl289/CS6241_proj/dataset/{}/{}.mtx".format(dataset, dataset), laplacian)
    file.close()

    return laplacian

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_graphzoom():
    dataset = dataset_str = 'cora'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/home/xl289/CS6241_proj/raw_data/ind.{}.{}".format('cora', names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("/home/xl289/CS6241_proj/raw_data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    laplacian = json2mtx(dataset)
    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    adj = nx.to_scipy_sparse_matrix(G, weight='wgt')

    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = process.sparse_mx_to_torch_sparse_tensor(adj)

    features = np.load("/home/xl289/CS6241_proj/dataset/cora/cora-feats.npy")
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # features = features.toarray()

    features = sp.csr_matrix(np.matrix(features))
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    int_labels = [
        (int(np.where(r == 1)[0][0]) if len(np.where(r==1)[0]) > 0 else -1)
        for r in labels]
    labels = torch.LongTensor(int_labels)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 2000)

    return adj, features, labels, idx_train, idx_val, idx_test


