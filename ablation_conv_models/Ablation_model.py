import torch.nn as nn


class Embedding_View(nn.Module):
    def __init__(self,max_len,embedding_size):
        super(Embedding_View, self).__init__()
        self.max_len=max_len
        self.embedding_size=embedding_size

    def forward(self,x):
        return x.view(-1, 1,self.max_len,self.embedding_size)

class Flatten(nn.Module):
    def __init__(self,start_dim=1):
        super(Flatten, self).__init__()
        self.start_dim=start_dim
    def forward(self,x):
        return x.flatten(start_dim=1).unsqueeze(1)

class TCPI_2_layer_Conv_Model(nn.Module):
    def __init__(self, embedding_size=200, drug_len=150, protein_len=1000, num_filter=64):
        super(TCPI_2_layer_Conv_Model, self).__init__()
        ###################################drug channel###################################
        # region
        self.drug_embedding = nn.Sequential(
            nn.Embedding(2048 + 1, embedding_size, padding_idx=0),
            Embedding_View(max_len=drug_len, embedding_size=embedding_size),
            nn.BatchNorm2d(num_features=1)
        )
        ###drug conv###
        self.drug_convs = nn.Sequential(
            ###drug conv-1###
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(5, 5)),
            nn.BatchNorm2d(num_features=num_filter),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
            ###drug conv-2###
            nn.Conv2d(in_channels=num_filter, out_channels=num_filter,
                      kernel_size=(5, 5)),
            nn.BatchNorm2d(num_features=num_filter),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
            ###drug Linear###
            Flatten(start_dim=1),
            nn.Linear(384, 256)
        )
        # endregion

        ###################################protein channel###################################
        # region

        self.protein_embedding = nn.Sequential(
            nn.Embedding(26, embedding_size, padding_idx=0),
            Embedding_View(max_len=protein_len, embedding_size=embedding_size),
            nn.BatchNorm2d(num_features=1)
        )
        ###protein conv###
        self.protein_convs = nn.Sequential(
            ###protein conv-1###
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(5, 5)),
            nn.BatchNorm2d(num_features=num_filter),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(12, 12)),
            ###protein conv-2###
            nn.Conv2d(in_channels=num_filter, out_channels=num_filter,
                      kernel_size=(5, 5)),
            nn.BatchNorm2d(num_features=num_filter),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(12, 12)),
            ###protein Linear###
            Flatten(start_dim=1),
            nn.Linear(384, 256)
        )
        # endregion

        ###################################Linear###################################
        # region
        self.Linear_Head=nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            # nn.Softmax(dim=1)
        )
        # endregion


    def forward(self, drug_indexVec, protein_indexVec):
        drug_feature=self.drug_embedding(drug_indexVec)
        drug_feature=self.drug_convs(drug_feature)

        protein_feature = self.protein_embedding(protein_indexVec)
        protein_feature = self.protein_convs(protein_feature)

        merged_vector = (drug_feature + protein_feature).squeeze()

        output=self.Linear_Head(merged_vector)
        return output

class TCPI_1_layer_Conv_Model(nn.Module):
        def __init__(self, embedding_size=200, drug_len=150, protein_len=1000, num_filter=16):
            super(TCPI_1_layer_Conv_Model, self).__init__()
            ###################################drug channel###################################
            # region
            self.drug_embedding = nn.Sequential(
                nn.Embedding(2048 + 1, embedding_size, padding_idx=0),
                Embedding_View(max_len=drug_len, embedding_size=embedding_size),
                nn.BatchNorm2d(num_features=1)
            )
            ###drug conv###
            self.drug_convs = nn.Sequential(
                ###drug conv-1###
                nn.Conv2d(in_channels=1, out_channels=num_filter,
                          kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=num_filter),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(32, 32)),
                Flatten(start_dim=1),
                nn.Linear(384, 256)
            )
            # endregion

            ###################################protein channel###################################
            # region

            self.protein_embedding = nn.Sequential(
                nn.Embedding(26, embedding_size, padding_idx=0),
                Embedding_View(max_len=protein_len, embedding_size=embedding_size),
                nn.BatchNorm2d(num_features=1)
            )
            ###protein conv###
            self.protein_convs = nn.Sequential(
                ###protein conv-1###
                nn.Conv2d(in_channels=1, out_channels=num_filter,
                          kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=num_filter),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=(64, 64)),
                ###protein Linear###
                Flatten(start_dim=1),
                nn.Linear(720, 256)
            )
            # endregion

            ###################################Linear###################################
            # region
            self.Linear_Head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                # nn.Softmax(dim=1)
            )
            # endregion

        def forward(self, drug_indexVec, protein_indexVec):
            drug_feature = self.drug_embedding(drug_indexVec)
            drug_feature = self.drug_convs(drug_feature)

            protein_feature = self.protein_embedding(protein_indexVec)
            protein_feature = self.protein_convs(protein_feature)

            merged_vector = (drug_feature + protein_feature).squeeze()

            output = self.Linear_Head(merged_vector)
            return output




if __name__ == '__main__':
    pass