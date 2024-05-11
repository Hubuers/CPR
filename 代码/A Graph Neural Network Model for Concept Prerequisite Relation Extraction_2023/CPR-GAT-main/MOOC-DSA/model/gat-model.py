import pdb
import numpy as np
import time

from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import KFold



import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.nn import GATConv, Linear




class GAT(torch.nn.Module):
    def __init__(self, in_channels=300, out_channels1=128, out_channels2=512):
        super(GAT,self).__init__()
        self.conv1 = GATConv(in_channels, out_channels1, add_self_loops=False)
        self.lin1 = Linear(-1, out_channels1)
        self.conv2 = GATConv(out_channels1, out_channels2, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class SiameseNet(nn.Module):
	def __init__(self):
		super(SiameseNet,self).__init__()
		self.fc_layer = nn.Linear(512, 64)
		self.relu_layer = nn.ReLU()
		self.classificaton_layer = nn.Linear(256, 1)
		self.sigmoid_layer = nn.Sigmoid()

	def forward(self, x1, x2):
		c1 = self.relu_layer(self.fc_layer(x1))
		c2 = self.relu_layer(self.fc_layer(x2))
		
		diff = torch.sub(c1, c2)
		multiply = torch.mul(c1, c2)

		v = torch.cat((c1, c2, diff, multiply), 1)
		pred_prob = self.sigmoid_layer(self.classificaton_layer(v))

		return pred_prob

		

def coo_format(arr):
	
	value = arr[arr!=0]
	value = value[:, None]
	idx = np.argwhere(arr != 0)

	coo = idx.T
	
	return torch.tensor(value), torch.tensor(coo)

def find_true_label(ccf, p_i):

	p_i_list = []
	ccf_list = []
	labels = []

	for each_str in p_i:
		id_pair = each_str[:-1].split(" ")
		p_i_list.append(tuple(map(int, id_pair)))

	for i in ccf.T:
		ccf_list.append(tuple(i))

	for each_pair in ccf_list:
		if each_pair in p_i_list:
			labels.append(1)
		else:
			labels.append(0)

	return labels



def main():

	with open("./MOOC-DSA/data/train-data-index.txt", 'r') as f1:
		train_data = f1.readlines()
	
	train_data = [tuple(map(int, sample.split())) for sample in train_data]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)

	df = np.loadtxt("./MOOC-DSA/feature/df.txt")
	cf = np.loadtxt("./MOOC-DSA/feature/cf.txt")
	dcf = np.loadtxt("./MOOC-DSA/feature/dcf.txt")
	ddf = np.loadtxt("./MOOC-DSA/feature/ddf.txt")
	ccf = np.loadtxt("./MOOC-DSA/feature/ccf.txt")

	data = HeteroData()

	data['concept'].x = torch.tensor(cf)
	data['concept'].num_nodes = cf.shape[0]
	data['document'].x = torch.tensor(df)
	data['document'].num_nodes = df.shape[0]

	dcf_value, coo_dcf = coo_format(dcf)
	ddf_value, coo_ddf = coo_format(ddf)
	ccf_value, coo_ccf = coo_format(ccf)

	data['document', 'contains', 'concept'].edge_index = coo_dcf
	data['document', 'related_to', 'document'].edge_index = coo_ddf
	data['concept', 'similar_to', 'concept'].edge_index = coo_ccf

	data['document', 'contains', 'concept'].edge_attr = dcf_value
	data['document', 'related_to', 'document'].edge_attr = ddf_value
	data['concept', 'similar_to', 'concept'].edge_attr = ccf_value
	
	print(data)
	data = data.to(device)
	
	homogeneous_data = data.to_homogeneous().to(device)
	print(homogeneous_data)
	
	
	epochs = 500
	batch_size  = 4
	target_names = ['class 0', 'class 1']
	kf = KFold(n_splits = 5)
	td = shuffle(train_data)
	
	for i_fold, (train_idx, test_idx) in enumerate(kf.split(td)):
		gat = GAT().to(device)
		siamesenet = SiameseNet().to(device)

		optimizer = torch.optim.Adam(list(gat.parameters())+list(siamesenet.parameters()), lr = 0.000001)
		criterion = nn.BCELoss()

		X_train = [td[each_idx] for each_idx in train_idx]
		X_test = [td[each_idx] for each_idx in test_idx]
		

		print("Fold number: {}".format(i_fold+1))
		print("-------------")
		print("Training!!!!")
		
		for epoch in range(epochs):
			X_train = np.array(shuffle(X_train))

			total_loss = 0
			pred_label = []
			gat.train()
			siamesenet.train()

			batch_idx = 0
			for i in range(X_train.shape[0] // batch_size):
			
				x = X_train[batch_idx * batch_size : batch_idx * batch_size + batch_size]
				batch_idx += 1

				c1, c2 = x[:, 0], x[:, 1]
				target = x[:, -1]

				optimizer.zero_grad()
				gat_output = gat(homogeneous_data.x.float(), homogeneous_data.edge_index)
				target = torch.tensor(target).to(device)
				preq_pred_prob = siamesenet(gat_output[c1], gat_output[c2])
				
				loss = criterion(preq_pred_prob, target[:, None].float())
				total_loss += loss
				loss.backward()
				optimizer.step()
			average_loss = (total_loss / len(td)).float()
			
			print("epoch: {}, average loss: {}".format(epoch, average_loss))
		print("Fold number: {}".format(i_fold+1))
		print("-------------")
		print("validation!!!!")	
		
		gat.eval()
		siamesenet.eval()
		batch_pred_prob_test = []
		batch_target_label_test = []
		batch_idx = 0
		X_test = np.array(X_test)
		for i in range(X_test.shape[0] // batch_size):

			x = X_test[batch_idx * batch_size : batch_idx * batch_size + batch_size]
			batch_idx += 1

			c1, c2 = x[:, 0], x[:, 1]
			target = x[:, -1]

			gat_output_test = gat(homogeneous_data.x.float(), homogeneous_data.edge_index)
			target = torch.tensor(target).to(device)
			batch_target_label_test.append(target.data.cpu().tolist())
			preq_pred_prob = siamesenet(gat_output_test[c1], gat_output_test[c2])
			batch_pred_prob_test.append(preq_pred_prob[:,0].data.cpu().tolist())
		
		target_label_test = [each_target for each_list in batch_target_label_test for each_target in each_list]
		pred_prob_test = [each_prob for each_list in batch_pred_prob_test for each_prob in each_list]
		pred_label_test = [1 if prob >= 0.5 else 0 for prob in pred_prob_test]

		print(classification_report(target_label_test, pred_label_test, target_names=target_names))


if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	total_time = end_time - start_time
	print(total_time)