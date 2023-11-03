from models.GCN import GCN, train_GCN, test_GCN
from models.MAGNET import MAGNET, train_MAGNET, test_MAGNET
from models.FCN import FCN, train, test
from models.MLP import MLP, train_MLP, test_MLP
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.data_processing import load_data, pre_process_data, split_data
from data.data_process_FCN import process_data_FCN, process_data_MLP
import torch
from scipy.stats import pearsonr
import numpy as np
import logging
import csv
import time
import pickle as pk
from scipy.optimize import linprog

def run(args):
    '''
    Load and pre-process data

    Synthetic toy data: 8 reactions and 12 metabolites
    Augmented E.coli data: 95 reactions and 72 metabolites
    '''
    if args.data == 'ecoli':
        out_path = 'data/E.coli_data/'
        bounds = 'aug_NN_features.pk'
        S_matrix = 'S_matrix.csv'
        bounds, solutions, S_matrix, RAG = load_data(out_path, bounds, args.solution, S_matrix)
        S_bool = S_matrix[S_matrix == 0].fillna(1).to_numpy()
        input_features = 2
        in_c = 95
        out_c = 95
        label_dim = 95
        kernel = 2
    
    elif args.data == 'toy':
        out_path = 'data/Toy_data/'
        bounds = '1_node_features.pickle'
        solutions = 'toy_FBA_solutions.pickle'
        RAG = 'toy_adjacency_matrix.csv'
        bounds, solutions, S_matrix, RAG = load_data(out_path, bounds, solutions, RAG_file=RAG)
        S_bool = S_matrix
        input_features = 1
        in_c = 8
        out_c = 8
        label_dim = 8
        kernel = 1

    data = pre_process_data(bounds, solutions, RAG, args.data)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Logger")
    ###########################################################################
    '''
    Run the models

    Input: Upper and lower bounds (v_{l} and v_{u})
    Output: Solutions/ Flux vector v

    Main method:
    FCN: Fully Connected MLP a.k.a Null Space Network that operates in the nullspace (WINNING METHOD!)

    Baselines:
    MAGNET: Magnetic Laplacian based Directed Graph Neural Network
    MLP: Vanilla MLP with no constraints
    GCN: GCN with no constraints

    Ground truth for the optimization problem:
    LINPROG: SciPy Optimize function
    '''
    
     #################---GCN---##########################
    if args.model == "GCN":
        pcc_scores = []
        for i in range(5):
            csv_file_name = "GCN"+str(i)+args.data+".csv"
            train_data, test_data = split_data(data, args.split, random_state=i) #Do train test split here
            model = GCN(input_features)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            mse_loss = torch.nn.MSELoss()
            
            for epoch in range(0,args.epochs):
                loss = train_GCN(model,train_data, optimizer, mse_loss)
                logger.log(logging.INFO,f'Epoch {epoch + 1}, Train Loss: {loss:.4f}')
                print(f'Epoch {epoch + 1}, Train Loss: {loss:.4f}')

            acc, preds, labels = test_GCN(model,test_data, mse_loss)
            file_contents = np.column_stack((preds, labels))
            with open(csv_file_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write headers for the two columns (optional)
                writer.writerow(["Predictions", "Labels"])
                # Write the data from the arrays to the CSV file
                writer.writerows(file_contents) 
            logger.log(logging.INFO,f'Test Error: {acc:.4f}')
            print(f'Test Error: {acc:.4f}')

            #Find PCC between predicted values and ground truth
            pcc, _ = pearsonr(labels, preds)
            pcc_scores.append(pcc)
            logger.log(logging.INFO,f'Pearson Correlation Coefficient: {pcc}')
            print("Pearson Correlation Coefficient:", pcc)
        
        mean_pcc = np.mean(pcc_scores)
        std_dev = np.std(pcc_scores)
        logger.log(logging.INFO,f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")
        print(f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")

    #################---LINPROG---##########################
    if args.model == "LINPROG":
        with open('data/bounds_vals.pk', 'rb') as b:
            bounds = pk.load(b)
        
        with open('data/c_vals.pk', 'rb') as c:
            c_lst = pk.load(c)
        runtime_lp = []
        
        for s, bounds in enumerate(bounds):
            c = c_lst[s]
            start_time_lp = time.time()
            res = linprog(c,A_ub=None, b_ub=None, A_eq = S_matrix.to_numpy(), b_eq = np.zeros(S_matrix.shape[0]),
                    bounds=bounds) #, method= method
            runtime_lp.append(time.time() - start_time_lp)
        
        logger.log(logging.INFO,f"Runtime: {np.mean(runtime_lp):.3f} ± {np.std(runtime_lp):.3f}")
        print("DONE!")

   #################---FCN---########################## 
    if args.model == "FCN":
        
        pcc_scores = []
        # runtime_scores = []
        for i in range(1):
            # start_time = time.time()
            csv_file_name = "NullSpaceNetwork"+str(i)+args.data+".csv"
            train_data, val_data, test_data, ns = process_data_FCN(data, S_matrix, randomseed=i)
            cuda_device = 0
            device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
            model = FCN(in_c, out_c, ns, label_dim, kernel, layer = 2, num_filter = 32, dropout=0).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
            for epoch in range(0,args.epochs):
                loss = train(model, train_data, val_data, optimizer, scheduler, epoch, args.epochs)
                logger.log(logging.INFO,loss)
                print(loss)
            
            acc, preds, labels, runtime = test(model, test_data)
            # end_time = time.time()

            logger.log(logging.INFO,f"Runtime: {np.mean(runtime):.3f} ± {np.std(runtime):.3f}")
            #Save outputs and solutions in a csv file
            file_contents = np.column_stack((preds, labels))
            with open(csv_file_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write headers for the two columns (optional)
                writer.writerow(["Predictions", "Labels"])
                # Write the data from the arrays to the CSV file
                writer.writerows(file_contents) 
            print(acc)
            logger.log(logging.INFO,acc)
            #Find PCC between predicted values and ground truth
            pcc, _ = pearsonr(labels, preds)
            pcc_scores.append(pcc)
            print("Pearson Correlation Coefficient:", pcc)
            # runtime_scores.append(time.time() - start_time)

        mean_pcc = np.mean(pcc_scores)
        std_dev = np.std(pcc_scores)
        logger.log(logging.INFO,f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")
        print(f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")

    
    #################---MAGNET---##########################
    if args.model == "MAGNET":
        pcc_scores = []
        for i in range(5):
            csv_file_name = "MAGNET"+str(i)+args.data+".csv"
            train_data, test_data = split_data(data, args.split, random_state=i) #Do train test split here
            model = MAGNET(input_features)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            mse_loss = torch.nn.MSELoss()
            for epoch in range(0,args.epochs):
                loss = train_MAGNET(model,train_data, optimizer, mse_loss, S_bool)
                logger.log(logging.INFO,f'Epoch {epoch + 1}, Train Loss: {loss:.4f}')
                print(f'Epoch {epoch + 1}, Train Loss: {loss:.4f}')
            acc, preds, labels = test_MAGNET(model,test_data, mse_loss)
            file_contents = np.column_stack((preds, labels))
            with open(csv_file_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write headers for the two columns (optional)
                writer.writerow(["Predictions", "Labels"])
                # Write the data from the arrays to the CSV file
                writer.writerows(file_contents) 
            logger.log(logging.INFO,f'Test Error: {acc:.4f}')
            print(f'Test Accuracy: {acc:.4f}')

            #Find PCC between predicted values and ground truth
            pcc, _ = pearsonr(labels, preds)
            pcc_scores.append(pcc)
            logger.log(logging.INFO,f'Pearson Correlation Coefficient: {pcc}')
            print("Pearson Correlation Coefficient:", pcc)
        
        mean_pcc = np.mean(pcc_scores)
        std_dev = np.std(pcc_scores)
        logger.log(logging.INFO,f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")
        print(f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")
    
    #################---MLP---##########################
    if args.model == "MLP":
        pcc_scores = []
        for i in range(5):
            csv_file_name = "MLP"+str(i)+args.data+".csv"
            train_data, val_data, test_data = process_data_MLP(data, S_matrix, randomseed=i)
            cuda_device = 0
            device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
            model = MLP(input_size=input_features, output_size=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(0,args.epochs):
                loss = train_MLP(model, train_data, val_data, optimizer, epoch, args.epochs)
                logger.log(logging.INFO,loss)
                print(loss)
            acc, preds, labels = test_MLP(model, test_data)
            file_contents = np.column_stack((preds, labels))
            with open(csv_file_name, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                # Write headers for the two columns (optional)
                writer.writerow(["Predictions", "Labels"])
                # Write the data from the arrays to the CSV file
                writer.writerows(file_contents) 
            print(acc)

            #Find PCC between predicted values and ground truth
            pcc, _ = pearsonr(labels, preds)
            pcc_scores.append(pcc)
            print("Pearson Correlation Coefficient:", pcc)
        
        mean_pcc = np.mean(pcc_scores)
        std_dev = np.std(pcc_scores)
        print(f"PCC: {mean_pcc:.3f} ± {std_dev:.3f}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN to compute flux rate')
    parser.add_argument('--model', type=str,
                        default='FCN',
                        help='Model choice (default:FCN)')
    parser.add_argument('--data', type=str, 
                        default='ecoli',
                        help='Data choice (default: E.coli data)')
    parser.add_argument('--solution', type=str,
                        default='mwclique',
                        help='Solution type i.e biomass or mwclique (default: mwclique)')
    parser.add_argument('--epochs', type=int,
                        default=100,
                        help='Number of epochs (default:100)')
    parser.add_argument('--lr',type = float,
                        default=0.01,
                        help='Learning rate (default:0.01)')
    parser.add_argument('--split', type=float,
                        default=0.2,
                        help='Train test split (default:0.2)'
                        )
    args = parser.parse_args()

    run(args)


