#!/usr/bin/env python3

import os
import sys
import ctypes
import struct
import inspect
import argparse
import urllib

import numpy as np
import torch
import dgl
from dgl.data import RedditDataset, CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from scipy.sparse import csr_matrix, coo_matrix

from scipy import sparse

# from matrixpart import PaToHMatrixPart as patoh

seed = 0

def permute_sparse_matrix(M, new_row_order=None, new_col_order=None):
	"""
	Reorders the rows and/or columns in a scipy sparse matrix 
		using the specified array(s) of indexes
		e.g., [1,0,2,3,...] would swap the first and second row/col.
	"""
	if new_row_order is None and new_col_order is None:
		return M
	
	new_M = M
	if new_row_order is not None:
		new_row_order = np.argsort(new_row_order)
		I = sparse.eye(M.shape[0]).tocoo()
		I.row = I.row[new_row_order]
		new_M = I.dot(new_M)
	if new_col_order is not None:
		new_col_order = np.argsort(new_col_order)
		I = sparse.eye(M.shape[1]).tocoo()
		I.col = I.col[new_col_order]
		new_M = new_M.dot(I)
	return new_M

def serialize_sparse_matrix(A, fname):
	try:
		A = csr_matrix(A)
		vtype = 'I' if A.shape[0] < ctypes.c_uint32(-1).value else 'Q'
		etype = 'I' if A.getnnz() < ctypes.c_uint32(-1).value else 'Q'
		info_line = ('4' if vtype == 'I' else '8') + ' ' + ('4' if vtype == 'I' else '8') + '\n'
		with open(fname, 'wb') as f:
			# f.write(info_line.encode("utf-8"))
			f.write('PIGO-CSR-v2'.encode("utf-8"))
			f.write(np.array([4, 4], dtype='uint8'))
			f.write(np.array([A.shape[0]], dtype='uint32' if vtype == 'I' else 'uint64'))
			f.write(np.array([A.nnz], dtype='uint32' if etype == 'I' else 'uint64'))
			f.write(np.array([A.shape[0], A.shape[0]], dtype='uint32' if vtype == 'I' else 'uint64'))
			# f.write(struct.pack(vtype + vtype, *A.shape))
			f.write(A.indptr.astype('uint32' if vtype == 'I' else 'uint64'))
			f.write(A.indices.astype('uint32' if etype == 'I' else 'uint64'))
			f.write(A.data.astype('float32')) # Change this, maybe add data type to the info line as well.
	except Exception as e:
		print("Error converting ", fname)
		raise e

def serialize_matrix(A, fname, dtype):
	try:
		A = np.array(A, dtype=dtype, copy=False, order='C')
		shape = np.array(list(A.shape), dtype='uint32')
		with open(fname, 'wb') as f:
			f.write(shape)
			f.write(A)
	except Exception as e:
		print("Error converting ", fname)
		raise e

def serialize_dataset(name, graph, features, labels, sets):
	global seed
	if seed != 0:
		name = os.path.join("permuted", name)
	try:
		os.mkdir(name)
	except FileExistsError as _: 
		pass

	if seed != 0:
		rng = np.random.default_rng(seed)
		p = rng.permutation(features.shape[0])
		graph = csr_matrix(graph)
		graph = permute_sparse_matrix(graph, p, p)
		features = features[p, :]
		labels = labels[p, :]
		sets = sets[p, :]

	serialize_sparse_matrix(graph, os.path.join(name, 'graph.bin'))
	serialize_matrix(features, os.path.join(name, 'features.bin'), 'float32')
	serialize_matrix(np.array(labels).reshape([-1, 1]), os.path.join(name, 'labels.bin'), 'uint32')
	serialize_matrix(np.array(sets).reshape([-1, 1]), os.path.join(name, 'sets.bin'), 'uint32')

def serialize_dgl_graph(g, name, P=8):
	N = g.number_of_nodes() + P - 1
	N -= N % P + g.number_of_nodes()

	g.add_nodes(N, {
		'label': torch.zeros([N], dtype=g.ndata['label'].dtype),
		'train_mask': torch.zeros([N], dtype=g.ndata['train_mask'].dtype),
		'val_mask': torch.zeros([N], dtype=g.ndata['val_mask'].dtype),
		'test_mask': torch.zeros([N], dtype=g.ndata['test_mask'].dtype),
		'feat': torch.zeros([N] + list(g.ndata['feat'].shape[1:]), dtype=g.ndata['feat'].dtype)
	})

	g = g.add_self_loop()

	labels = g.ndata['label'].reshape([-1, 1])

	sets = np.zeros([g.number_of_nodes(), 1], dtype='uint32')
	sets[g.ndata['train_mask']] = 0
	sets[g.ndata['val_mask']] = 1
	sets[g.ndata['test_mask']] = 2

	K = g.ndata['feat'].shape[-1] + P - 1
	K -= K % P + g.ndata['feat'].shape[-1]
	g.ndata['feat'] = torch.cat((g.ndata['feat'], torch.zeros(list(g.ndata['feat'].shape[:-1]) + [K], dtype=g.ndata['feat'].dtype)), -1)

	serialize_dataset(name, g.adjacency_matrix(scipy_fmt="csr"), g.ndata['feat'], labels, sets)

def serialize_dgl_dataset(data):
	serialize_dgl_graph(data[0], data.name)

def serialize_graph(g, name, num_feats, num_labels):
		g = dgl.DGLGraph(g)
		N = g.number_of_nodes()
		g.ndata['feat'] = torch.zeros([N, num_feats], dtype=torch.float32)
		g.ndata['train_mask'] = torch.ones([N], dtype=torch.bool)
		g.ndata['val_mask'] = torch.zeros([N], dtype=torch.bool)
		g.ndata['test_mask'] = torch.zeros([N], dtype=torch.bool)
		g.ndata['label'] = torch.ones([N], dtype=torch.int32) * (num_labels - 1)

		serialize_dgl_graph(g, name)

def serialize_ogb_dataset(name, rnd_feat=False):
	dataset = DglNodePropPredDataset(name)
	g, label = dataset[0]
	if rnd_feat:
		serialize_graph(g, name, 128, 48)
	else:
		N = g.number_of_nodes()
		g.ndata['label'] = label.reshape(-1)
		split_idx = dataset.get_idx_split()
		g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'] = torch.zeros(N, dtype=torch.bool).index_fill_(0, split_idx['train'], True), torch.zeros(N, dtype=torch.bool).index_fill_(0, split_idx['valid'], True), torch.zeros(N, dtype=torch.bool).index_fill_(0, split_idx['test'], True)

		serialize_dgl_graph(g, name)

def serialize_toy_example():
	graph = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
	labels = [0, 1, 0, 1]
	sets = [0, 0, 1, 2]
	features = [[0, 1], [1, 0], [0, 1], [1, 0]]

	serialize_dataset('toyA', graph, features, labels, sets)

	graph = [[0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]
	labels = [0, 1, 0, 1]
	sets = [0, 0, 1, 2]
	features = [[0, 1], [1, 0], [0, 1], [1, 0]]

	serialize_dataset('toyB', graph, features, labels, sets)

import ssgetpy
from scipy.io import mmread
def serialize_ss_dataset(name, num_features, num_labels):
	dataset = ssgetpy.search(name)
	dataset[0].download(extract=True, destpath="/tmp")
	graph = mmread("/tmp/{}/{}.mtx".format(name, name))
	serialize_graph(graph, name, num_features, num_labels)

def serialize_ss():
	dataset = ssgetpy.search(group="SNAP", rowbounds=(2000000, 10000000))
	print(len(dataset))
	for d in dataset:
		serialize_ss_dataset(d.name, 128, 48)

def download_file(url):
	name = url.rsplit('/', 1)[-1]
	filename = os.path.join('./', name)
	if not os.path.isfile(filename):
		urllib.urlretrieve(url, filename)
	return filename

def download_matrix(url):
	fn = download_file(url)
	return csr_matrix(mmread(fn))

def proteins():
	return download_matrix('https://portal.nersc.gov/project/m1982/HipMCL/subgraphs/subgraph3_iso_vs_iso_30_70length_ALL.m100.oneindexed.mtx')

def read_matrix(name):
	try:
		with open(os.path.join(name, 'graph.bin'), 'rb') as f:
			s = f.read(11)
			assert(s == b'PIGO-CSR-v2')
			s = f.read(2)
			assert(s[0] == 4 and s[1] == 4)
			(_, nnz, N, M) = struct.unpack('IIII', f.read(16))
			indptr = np.frombuffer(f.read((N + 1) * 4), dtype='uint32')
			indices = np.frombuffer(f.read(nnz * 4), dtype='uint32')
			data = np.frombuffer(f.read(nnz * 4), dtype='float32')
			g = csr_matrix((data, indices, indptr), shape=(N, M))
	except:
		print('ex')
		if name == 'reddit':
			g = RedditDataset()[0]
			g = g.adj(scipy_fmt='csr')
		elif name == 'cora_v2':
			g = CoraGraphDataset()[0]
			g = g.adj(scipy_fmt='csr')
		elif name == 'ogbn-arxiv' or name == 'ogbn-products':
			g, _ = DglNodePropPredDataset(name)[0]
			g = g.adj(scipy_fmt='csr')
		elif name == 'proteins':
			g = proteins()
		else:
			dataset = ssgetpy.search(name)
			dataset[0].download(extract=True, destpath="/tmp")
			g = mmread("/tmp/{}/{}.mtx".format(name, name))
	
	print(g.shape, g.nnz)

	return g
		
def compute_comm(args):
	g = read_matrix(args.graph)

	pv = None

	if len(args.permutation_file) > 0:
		if args.permutation_file == "patoh":
			# pv = patoh(g, args.number_of_parts, 'RWS')[1] - 1
			p = np.argsort(pv)
		else:
			with open(args.permutation_file) as f:
				a = f.readlines()
			p = np.array([int(i) for i in a.split()], dtype=np.uint64)
	else:
		if args.seed == 0:
			p = np.arange(g.shape[0], dtype=np.uint64)
		else:
			rng = np.random.default_rng(seed)
			p = rng.permutation(g.shape[0])
	
	g = permute_sparse_matrix(g, p, p)

	N = args.number_of_parts

	if pv is None:
		p = np.array([i * g.shape[0] // N for i in range(N + 1)], dtype=np.uint64)
	else:
		p = np.concatenate(([0], np.cumsum(np.bincount(pv))))

	print(p)
	print(np.diff(p))

	L = np.zeros([N, N], dtype=np.uint64)

	for i in range(N):
		mask = np.zeros(g.shape[1], dtype=np.bool8)
		mask[g.indices[g.indptr[p[i]]: g.indptr[p[i + 1]]]] = True
		for j in range(N):
			L[i, j] = np.sum(mask[p[j]: p[j + 1]])
	
	print(L)

def coo_to_csr(filename):
	row = []
	col = []
	seen = set()
	with open(filename, 'r') as f:
		lines = f.readlines()
		for line in lines:
			r, c = line.strip().split(',')
			if (r, c) in seen:
				continue
			seen.add((r, c))
			row.append(int(r) - 1)
			col.append(int(c) - 1)
		row = np.array(row)
		col = np.array(col)
		data = np.ones(row.shape)
		n = max(np.max(row), np.max(col)) + 1
		coo = coo_matrix((data, (row, col)), shape=(n, n))
		return coo.tocsr()
	return None
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--seed', type=int, help='seed for random permutation', default=0)
	parser.add_argument('-p', '--permutation-file', type=str, help='permutation file', default='patoh')
	parser.add_argument('-c', '--communication', action='store_true')
	parser.add_argument('-g', '--graph', type=str, help='graph name')
	parser.add_argument('-n', '--number-of-parts', type=int, default=8)
	args = parser.parse_args()

	seed = args.seed

	if args.communication:
		compute_comm(args)
	else:
		serialize_dgl_dataset(RedditDataset())
		# serialize_dgl_dataset(CoraGraphDataset())
		# serialize_ogb_dataset('ogbn-arxiv')
		# serialize_ogb_dataset('ogbn-products')
		# serialize_ogb_dataset('ogbn-papers100M')
		# serialize_toy_example()
		# serialize_ss_dataset("as-caida", 128, 48)
		# serialize_ss()
		# serialize_graph(read_matrix('proteins'), 'proteins', 128, 256)
		# for i in [32]:
		# 	for j in range(1, 11):
		# 		serialize_graph(coo_to_csr('{}_{}x_random.csv'.format(j, i)), '{}_{}x_random'.format(j, i), 512, 40)
		# serialize_graph(coo_to_csr('1_128x_random.csv'.format(j, i)), '1_128x_random'.format(j, i), 512, 40)

