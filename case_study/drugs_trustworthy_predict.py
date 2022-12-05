import argparse

import torch
import numpy as np

from MyUtils.helpers import get_device
from MyUtils.losses import relu_evidence
from prettytable import PrettyTable


drugs_name_list = ["Abacavir", "Aciclovir", "Adefovir", "Amantadine", "Amprenavir", "Arbidol", "Atazanavir",
                   "Baloxavir", "Bictegravir", "Emtricitabine", "Tenofovir", "Boceprevir", "Cidofovir", "Cobicistat",
                   "Lamivudine", "Zidovudine", "Daclatasvir", "Darunavir", "Delavirdine", "Descovy", "Didanosine",
                   "Docosanol", "Dolutegravir", "Doravirine", "Edoxudine", "Efavirenz", "Elvitegravir", "Enfuvirtide",
                   "Entecavir", "Etravirine", "Famciclovir", "Fosamprenavir", "Foscarnet", "Ganciclovir", "Ibacitabine",
                   "Idoxuridine", "Imiquimod", "Imunovir", "Indinavir", "Inosine", "Letermovir", "Lopinavir",
                   "Loviride", "Maraviroc", "Methisazone", "Moroxydine", "Nelfinavir", "Nevirapine", "Nitazoxanide",
                   "Ritonavir", "Oseltamivir", "Penciclovir", "Peramivir", "Pleconaril", "Podophyllotoxin",
                   "Glecaprevir", "Grazoprevir", "Pyrimidine", "Raltegravir", "Remdesivir", "Ribavirin", "Rilpivirine",
                   "Rimantadine", "Saquinavir", "Simeprevir", "Sofosbuvir", "Stavudine", "Chloroquine", "Telaprevir",
                   "Telbivudine", "Tenofovir_disoproxil", "Tipranavir", "Trifluridine", "Tromantadine", "Valacyclovir",
                   "Valganciclovir", "Vicriviroc", "Vidarabine", "Taribavirin", "Zalcitabine", "Zanamivir",
                   "Hydroxychloroquine", " Amoxicillin", "Penicillin", "Aspirin"]

if __name__ == '__main__':
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument(
        "--num_classes", default=2, type=int, help="num_classes."
    )

    args = parser.parse_args()
    # Use the model pre-trained on C.elegans to predict
    model = torch.load('model_trained_on_Celegans.pth')
    device = get_device()
    model = model.to(device)

    # Obtain the encoding vector of C-P pairs to be predicted
    IndexVec_data = np.load("existing_drugs_3CLPro_pair_IndexVec_data.npy", allow_pickle=True)

    # Save the predicted results
    table_data = []
    interaction_func = lambda x: "YES" if x > 0.5 else "NO"

    for i in range(len(IndexVec_data)):
        drug_indexVec, protein_indexVec, label = IndexVec_data[i]
        # Convert the data dimensions "drug_indexVec:(150,) ->(1,150) and protein_indexVec:(1000,)->(1,1000)" to input into the model
        drug_indexVec, protein_indexVec = drug_indexVec[np.newaxis, :], protein_indexVec[np.newaxis, :]
        drug_indexVec, protein_indexVec = torch.tensor(drug_indexVec).to(device), torch.tensor(protein_indexVec).to(
            device)
        outputs = model(drug_indexVec.long(), protein_indexVec.long())
        _, preds = torch.max(outputs, 0)
        # Convert the network output to non-negative values (i.e., evidence)
        evidence = relu_evidence(outputs)
        belief_masses = evidence + 1

        # Obtain the prediction probability of positive category Using the Eq. (4) in the paper
        prob = torch.softmax(belief_masses, dim=0)[1].item()

        # Probability>50% is considered as positive interaction
        interaction=interaction_func(prob)
        # Calculate the size of uncertainty using the Eq. (3) in the paper
        u = (args.num_classes / torch.sum(belief_masses, dim=0, keepdim=True)).view(-1).item()
        table_data.append([drugs_name_list[i], interaction,prob, u])

    table_data = np.array(table_data)
    # Sorting according to the size of Probability
    probs = table_data[:, 2]
    probs = probs.astype(np.float64)
    index = np.argsort(-probs)
    table_data = table_data[index]

    table = PrettyTable(['Rank', 'Drug_Name', 'Interaction','Probability', 'Uncertainty'])

    for i in range(len(table_data)):
        drug_name=table_data[i][0]
        interaction=table_data[i][1]
        probability=table_data[i][2]
        uncertainty=table_data[i][3]
        table.add_row([i+1, drug_name,interaction,probability,uncertainty])
    print(table)
    pass
