import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances

from alice.dataloader.data_utils import *


class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K
        self.train_set, self.train_dataloader, _, self.val_dataloader = get_validation_dataloader(args)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        n_data = len(self.train_set)
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                print('validation | training dataset --- index: {0}'.format(batch_idx))
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)
                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()
            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                print('validation | validation dataset --- index: {0}'.format(batch_idx))
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                features = self.model(inputs.to(self.device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()


class NCMValidation(object):
    def __init__(self, args, model):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.train_set, self.train_dataloader, _, self.val_dataloader = get_validation_dataloader(args)

    def _retrieval(self):
        """Extract features from validation split and search on train split features."""
        cls_wise_feature_prototype = []
        avg_cls = []
        embedding_list = []
        label_list = []
        validation_embedding_list = []
        validation_label_list = []

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        # --- using training data to generate average feature embedding for each class ---
        print('acquiring class-wise feature prototype from training data ...')
        with torch.no_grad():
            tqdm_gen = tqdm(self.train_dataloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = self.model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # generate the average feature with all data
        for index in range(self.args.base_class):
            class_index = (label_list == index).nonzero()
            embedding_this = embedding_list[class_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0, keepdims=True).cuda()
            cls_wise_feature_prototype.append(embedding_this)
            avg_cls.append(index)

        for i in range(len(cls_wise_feature_prototype)):
            cls_wise_feature_prototype[i] = cls_wise_feature_prototype[i].view(-1)
        proto_list = torch.stack(cls_wise_feature_prototype, dim=0).cpu()
        proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=-1)

        # --- acquire feature for each validation data ---
        print('acquiring feature prototype for testing data ...')
        with torch.no_grad():
            tqdm_gen = tqdm(self.val_dataloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = self.model(data)
                validation_embedding_list.append(embedding.cpu())
                validation_label_list.append(label.cpu())
        validation_embedding_list = torch.cat(validation_embedding_list, dim=0).cpu()
        validation_embedding_list = torch.nn.functional.normalize(validation_embedding_list, p=2, dim=-1)

        validation_label_list = torch.cat(validation_label_list, dim=0).cpu()

        # --- calculate the cosine similarity for each validation data ---
        # metric: euclidean, cosine, l2, l1
        pairwise_distance = pairwise_distances(np.asarray(validation_embedding_list), np.asarray(proto_list), metric='cosine')
        prediction_result = np.argmin(pairwise_distance, axis=1)

        validation_label_list = np.asarray(validation_label_list)
        top1 = np.sum(prediction_result == validation_label_list) / float(len(validation_label_list))
        return top1

    def eval(self):
        return self._retrieval()
