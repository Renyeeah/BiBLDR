import torch
import torch.nn as nn
import numpy as np
import os

from utils import get_batch_from_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AlignLayer(nn.Module):
    def __init__(self, args, embedding_dim):
        super().__init__()
        self.args = args
        self.embedding_dim = embedding_dim
        self.input_dim = self.args.promote_embedding_dim + self.embedding_dim

        self.network = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=int(self.input_dim*0.75)),
            nn.BatchNorm1d(num_features=int(self.input_dim*0.75)),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=int(self.input_dim*0.75), out_features=int(self.input_dim/2)),
            nn.BatchNorm1d(num_features=int(self.input_dim/2)),
            nn.ReLU(),
        )

    def forward(self, input):
        output = self.network(input)
        return output

class FusionNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.network = nn.Sequential(
            nn.Linear(in_features=self.args.promote_embedding_dim*2, out_features=int(self.args.promote_embedding_dim*1.5)),
            nn.BatchNorm1d(num_features=int(self.args.promote_embedding_dim*1.5)),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=int(self.args.promote_embedding_dim*1.5), out_features=self.args.promote_embedding_dim),
            nn.BatchNorm1d(num_features=self.args.promote_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            output = self.network(input)
            outputs.append(output)
        return outputs

class dimensionality_reduction_layer(nn.Module):
    def __init__(self, args, len_id):
        super().__init__()
        self.args = args
        self.result_dimension = self.args.promote_embedding_dim - int(np.sqrt(len_id))

        self.network = nn.Sequential(
            nn.Linear(in_features=self.args.promote_embedding_dim, out_features=int(0.9 * self.args.promote_embedding_dim)),
            nn.BatchNorm1d(num_features=int(0.9 * self.args.promote_embedding_dim)),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=int(0.9 * self.args.promote_embedding_dim), out_features=self.result_dimension),
            nn.BatchNorm1d(num_features=self.result_dimension),
            nn.ReLU(),
        )

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            output = self.network(input)
            outputs.append(output)
        return outputs

class MHALayer(nn.Module):
    def __init__(self, seq_len, args):
        super().__init__()
        self.args = args
        self.dim = self.args.promote_embedding_dim
        self.len_behavior = seq_len

        self.MHAttention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.args.num_heads, dropout=self.args.dropout)
        self.Dropout_1 = nn.Dropout(self.args.dropout)
        self.Dropout_2 = nn.Dropout(self.args.dropout)
        self.BatchNorm_1 = nn.BatchNorm1d(num_features=self.len_behavior)
        self.BatchNorm_2 = nn.BatchNorm1d(num_features=self.len_behavior)
        self.dense_layer = nn.Linear(in_features=self.len_behavior*self.dim, out_features=self.len_behavior*self.dim)

    def forward(self, input):
        multihead, _ = self.MHAttention(query=input, key=input, value=input)
        drop_multihead = self.Dropout_1(multihead)
        norm_multihead = self.BatchNorm_1(input + drop_multihead)
        activate_multihead = nn.functional.leaky_relu(norm_multihead)
        dense_feature = self.dense_layer(activate_multihead.view(activate_multihead.size(0), -1))
        dense_feature = dense_feature.reshape(dense_feature.size(0), self.len_behavior, self.dim)
        drop_dense = self.Dropout_2(dense_feature)
        output = self.BatchNorm_2(multihead + drop_dense)
        return output

class MLP_score(nn.Module):
    def __init__(self, embedding_dim_drug, embedding_dim_disease, args):
        super().__init__()
        self.args = args
        self.len_drug_feature = int((self.args.promote_embedding_dim + embedding_dim_drug)/2)
        self.len_drug_behavior = self.args.seqlen_drug_behavior*self.args.promote_embedding_dim
        self.len_disease_feature = int((self.args.promote_embedding_dim + embedding_dim_disease)/2)
        self.len_disease_behavior = self.args.seqlen_disease_behavior*self.args.promote_embedding_dim
        self.dim_input = self.len_drug_feature + self.len_drug_behavior + self.len_disease_feature + self.len_disease_behavior
        self.network = nn.Sequential(
            nn.Linear(in_features=self.dim_input, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=2048, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, input):
        output = self.network(input)
        return output

class DrugModel(nn.Module):
    def __init__(self, len_drug_id, len_dis_id, similarity_matrix, args):
        super().__init__()
        self.len_drug_id = len_drug_id
        self.len_dis_id = len_dis_id
        self.args = args
        self.similarity_matrix_drug = torch.tensor(similarity_matrix[0])
        self.similarity_matrix_drug = self.similarity_matrix_drug.to(device).type(torch.float)
        self.similarity_matrix_disease = torch.tensor(similarity_matrix[1])
        self.similarity_matrix_disease = self.similarity_matrix_disease.to(device).type(torch.float)

        self.embedding_dim_drug = int(np.sqrt(self.len_drug_id))
        self.embedding_drug = nn.Embedding(num_embeddings=self.len_drug_id, embedding_dim=self.embedding_dim_drug)
        self.embedding_dim_dis = int(np.sqrt(self.len_dis_id))
        self.embedding_disease = nn.Embedding(num_embeddings=self.len_dis_id, embedding_dim=self.embedding_dim_dis)

        # load promotes
        self.promotes_drug = torch.load(os.path.join('Stage1', self.args.dataset, str(args.promote_embedding_dim), 'promotes_drug.pt'))
        self.promotes_disease = torch.load(os.path.join('Stage1', self.args.dataset, str(args.promote_embedding_dim), 'promotes_disease.pt'))

        self.AlignLayer_drug = AlignLayer(self.args, self.embedding_dim_drug)
        self.AlignLayer_disease = AlignLayer(self.args, self.embedding_dim_dis)
        self.FusionNet_ChemS = FusionNet(self.args)
        self.FusionNet_DoS = FusionNet(self.args)
        self.dimensionality_reduction_layer_drug = dimensionality_reduction_layer(self.args, self.len_drug_id)
        self.dimensionality_reduction_layer_disease = dimensionality_reduction_layer(self.args, self.len_dis_id)
        self.TFLayer = MHALayer(self.args.seqlen_drug_behavior+self.args.seqlen_disease_behavior, self.args)
        self.MLP = MLP_score(self.embedding_dim_drug, self.embedding_dim_dis, self.args)

    def extract_drug_feature(self, drug_id):
        drug_promotes = get_batch_from_list(self.promotes_drug, drug_id)
        drug_promotes = torch.stack(drug_promotes)

        embedding_drug_id = self.embedding_drug(drug_id)

        input = torch.cat((drug_promotes, embedding_drug_id), dim=1)
        output = self.AlignLayer_drug(input)
        return output

    def extract_disease_feature(self, disease_id):
        disease_promotes = get_batch_from_list(self.promotes_disease, disease_id)
        disease_promotes = torch.stack(disease_promotes)

        embedding_disease_id = self.embedding_disease(disease_id)

        input = torch.cat((disease_promotes, embedding_disease_id), dim=1)
        output = self.AlignLayer_disease(input)
        return output

    def extract_behavior_drug(self, batch_drug_behavior, batch_drug_id):
        batch_promotes_behavior = torch.stack([torch.stack(get_batch_from_list(self.promotes_drug, drug_behavior)) for drug_behavior in batch_drug_behavior])
        batch_similarity = torch.stack([torch.stack(get_batch_from_list(self.similarity_matrix_drug[drug_id], drug_behavior)) for drug_behavior, drug_id in zip(batch_drug_behavior, batch_drug_id)])
        batch_similarity = batch_similarity.unsqueeze(-1)
        batch_sim_vector = batch_similarity.expand(-1, -1, batch_promotes_behavior.size(-1))
        batch_fusion_seq = torch.cat((batch_promotes_behavior, batch_sim_vector), dim=-1)
        batch_fusion_seq = batch_fusion_seq.permute(1, 0, 2)

        fusion_features = self.FusionNet_ChemS(batch_fusion_seq)
        fusion_features = torch.stack(fusion_features)
        dimension_reduce_features = self.dimensionality_reduction_layer_drug(fusion_features)
        dimension_reduce_features = torch.stack(dimension_reduce_features)
        dimension_reduce_features = dimension_reduce_features.permute(1, 0, 2)

        embedding_drug_behavior = self.embedding_drug(batch_drug_behavior)
        feature = torch.cat((dimension_reduce_features, embedding_drug_behavior), dim=2)
        return feature

    def extract_behavior_disease(self, batch_disease_behavior, batch_disease_id):
        batch_promotes_behavior = torch.stack([torch.stack(get_batch_from_list(self.promotes_disease, disease_behavior)) for disease_behavior in batch_disease_behavior])
        batch_similarity = torch.stack([torch.stack(get_batch_from_list(self.similarity_matrix_disease[disease_id], disease_behavior)) for disease_behavior, disease_id in zip(batch_disease_behavior, batch_disease_id)])
        batch_similarity = batch_similarity.unsqueeze(-1)
        batch_sim_vector = batch_similarity.expand(-1, -1, batch_promotes_behavior.size(-1))
        batch_fusion_seq = torch.cat((batch_promotes_behavior, batch_sim_vector), dim=-1)
        batch_fusion_seq = batch_fusion_seq.permute(1, 0, 2)

        fusion_features = self.FusionNet_DoS(batch_fusion_seq)
        fusion_features = torch.stack(fusion_features)
        dimension_reduce_features = self.dimensionality_reduction_layer_disease(fusion_features)
        dimension_reduce_features = torch.stack(dimension_reduce_features)
        dimension_reduce_features = dimension_reduce_features.permute(1, 0, 2)

        embedding_disease_behavior = self.embedding_disease(batch_disease_behavior)
        feature = torch.cat((dimension_reduce_features, embedding_disease_behavior), dim=2)
        return feature

    def forward(self, batch):
        drug_id = batch[0].to(device)
        drug_behavior = batch[1].to(device)
        drug_rating = batch[2].to(device).unsqueeze(-1)
        disease_id = batch[3].to(device)
        disease_behavior = batch[4].to(device)
        disease_rating = batch[5].to(device).unsqueeze(-1)

        drug_feature = self.extract_drug_feature(drug_id)
        drug_feature_behavior = self.extract_behavior_drug(drug_behavior, drug_id)
        calculate_drug_rating = torch.exp(self.args.rating_T*drug_rating)
        input_TF_drug = torch.mul(calculate_drug_rating, drug_feature_behavior)

        disease_feature = self.extract_disease_feature(disease_id)
        disease_feature_behavior = self.extract_behavior_disease(disease_behavior, disease_id)
        calculate_disease_rating = torch.exp(self.args.rating_T*disease_rating)
        input_TF_disease = torch.mul(calculate_disease_rating, disease_feature_behavior)

        input_TF = torch.cat((input_TF_drug, input_TF_disease), dim=1)
        # output_TF = self.TFLayer(input_TF)

        input_MLP = torch.cat((drug_feature, input_TF.view(input_TF.size(0), -1), disease_feature), dim=1)
        result = self.MLP(input_MLP)
        return result

class MLP_score_denova(nn.Module):
    def __init__(self, embedding_dim_drug, embedding_dim_disease, args):
        super().__init__()
        self.args = args
        self.len_drug_feature = int((self.args.promote_embedding_dim + embedding_dim_drug)/2)
        self.len_drug_behavior = self.args.seqlen_drug_behavior*self.args.promote_embedding_dim
        self.len_disease_feature = int((self.args.promote_embedding_dim + embedding_dim_disease)/2)
        self.dim_input = self.len_drug_feature + self.len_drug_behavior + self.len_disease_feature
        self.network = nn.Sequential(
            nn.Linear(in_features=self.dim_input, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=2048, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, input):
        output = self.network(input)
        return output

class DrugModel_denova(nn.Module):
    def __init__(self, len_drug_id, len_dis_id, similarity_matrix, args):
        super().__init__()
        self.len_drug_id = len_drug_id
        self.len_dis_id = len_dis_id
        self.args = args
        self.similarity_matrix_drug = torch.tensor(similarity_matrix[0])
        self.similarity_matrix_drug = self.similarity_matrix_drug.to(device).type(torch.float)
        self.similarity_matrix_disease = torch.tensor(similarity_matrix[1])
        self.similarity_matrix_disease = self.similarity_matrix_disease.to(device).type(torch.float)

        self.embedding_dim_drug = int(np.sqrt(self.len_drug_id))
        self.embedding_drug = nn.Embedding(num_embeddings=self.len_drug_id, embedding_dim=self.embedding_dim_drug)
        self.embedding_dim_dis = int(np.sqrt(self.len_dis_id))
        self.embedding_disease = nn.Embedding(num_embeddings=self.len_dis_id, embedding_dim=self.embedding_dim_dis)

        # load prototypes
        self.promotes_drug = torch.load(os.path.join('Stage1', self.args.dataset, 'promotes_drug.pt'))
        self.promotes_disease = torch.load(os.path.join('Stage1', self.args.dataset, 'promotes_disease.pt'))

        self.AlignLayer_drug = AlignLayer(self.args, self.embedding_dim_drug)
        self.AlignLayer_disease = AlignLayer(self.args, self.embedding_dim_dis)
        self.FusionNet_ChemS = FusionNet(self.args)
        self.dimensionality_reduction_layer_drug = dimensionality_reduction_layer(self.args, self.len_drug_id)
        self.MHALayer_drug = MHALayer(self.args.seqlen_drug_behavior, self.args)
        self.MLP = MLP_score_denova(self.embedding_dim_drug, self.embedding_dim_dis, self.args)

    def extract_drug_feature(self, drug_id):
        drug_promotes = get_batch_from_list(self.promotes_drug, drug_id)
        drug_promotes = torch.stack(drug_promotes)

        embedding_drug_id = self.embedding_drug(drug_id)

        input = torch.cat((drug_promotes, embedding_drug_id), dim=1)
        output = self.AlignLayer_drug(input)
        return output

    def extract_disease_feature(self, disease_id):
        disease_promotes = get_batch_from_list(self.promotes_disease, disease_id)
        disease_promotes = torch.stack(disease_promotes)

        embedding_disease_id = self.embedding_disease(disease_id)

        input = torch.cat((disease_promotes, embedding_disease_id), dim=1)
        output = self.AlignLayer_disease(input)
        return output

    def extract_behavior_drug(self, batch_drug_behavior, batch_drug_id):
        batch_promotes_behavior = torch.stack([torch.stack(get_batch_from_list(self.promotes_drug, drug_behavior)) for drug_behavior in batch_drug_behavior])
        batch_similarity = torch.stack([torch.stack(get_batch_from_list(self.similarity_matrix_drug[drug_id], drug_behavior)) for drug_behavior, drug_id in zip(batch_drug_behavior, batch_drug_id)])
        batch_similarity = batch_similarity.unsqueeze(-1)
        batch_sim_vector = batch_similarity.expand(-1, -1, batch_promotes_behavior.size(-1))
        batch_fusion_seq = torch.cat((batch_promotes_behavior, batch_sim_vector), dim=-1)
        batch_fusion_seq = batch_fusion_seq.permute(1, 0, 2)

        fusion_features = self.FusionNet_ChemS(batch_fusion_seq)
        fusion_features = torch.stack(fusion_features)
        dimension_reduce_features = self.dimensionality_reduction_layer_drug(fusion_features)
        dimension_reduce_features = torch.stack(dimension_reduce_features)
        dimension_reduce_features = dimension_reduce_features.permute(1, 0, 2)

        embedding_drug_behavior = self.embedding_drug(batch_drug_behavior)
        feature = torch.cat((dimension_reduce_features, embedding_drug_behavior), dim=2)
        return feature

    def forward(self, batch):
        drug_id = batch[0].to(device)
        drug_behavior = batch[1].to(device)
        drug_rating = batch[2].to(device).unsqueeze(-1)
        disease_id = batch[3].to(device)

        drug_feature = self.extract_drug_feature(drug_id)
        drug_feature_behavior = self.extract_behavior_drug(drug_behavior, drug_id)
        calculate_drug_rating = torch.exp(self.args.rating_T*drug_rating)
        input_transformer_drug = torch.mul(calculate_drug_rating, drug_feature_behavior)
        output_TF_drug = self.MHALayer_drug(input_transformer_drug)

        disease_feature = self.extract_disease_feature(disease_id)

        input_MLP = torch.cat((drug_feature, output_TF_drug.view(output_TF_drug.size(0), -1), disease_feature), dim=1)
        result = self.MLP(input_MLP)
        return result

class stage1_encoder(nn.Module):
    def __init__(self, len_id, args):
        super().__init__()
        self.len_id = len_id
        self.args = args

        self.network = nn.Sequential(
            nn.Linear(in_features=self.len_id, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(in_features=512, out_features=self.args.promote_embedding_dim),
            nn.BatchNorm1d(self.args.promote_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, input_1, input_2):
        output_1 = self.network(input_1)
        output_2 = self.network(input_2)
        return output_1, output_2



