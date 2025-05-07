from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from models.model_motcat import MCATPathwaysMotCat
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually.
    Print useful info (num fold and init databases, lenght of train and val ds), saves datasets of split an unique matrix in results directory.

    Args:
        - datasets : tuple
        - cur : Int     (num of fold)
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    # save datasets as unique matrix in directory of results
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function.
    Prints what it's doing, checks that in the args it's indicated 'nullSurvLoss' and calls it.
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss (or NLLRankSurvLoss)
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)  # loss function taken from other repository, little documentation in this one
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    Get option for optimizer from the args and get corresponding one
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    r'''
    Checks type of path for omics (xena, hallmarks, combine, multi) given as args to set input dimension of omics.
    Checks for the modality:
        (omics baseline)
        - (comparison) "mlp_per_path":  aggregate the genes into pathways using the pathway composition ds
        - "omics":  multilayer perceptron MLP to handle tabular omics data
        - (comparison) "snn":  genomics self normalizing network to handle tabular omics data
        (unimodal and multimodal baseline)
        - (comparison) ["abmil_wsi", "abmil_wsi_pathways"]: Attention MIL (multiple instance learning) for the unimodal (WSI only) and multimodal setting (pathways + WSI)
        - (comparison) ["deepmisl_wsi", "deepmisl_wsi_pathways"]: DeepMISL for unimodal (WSI only) and multimodal (WSI + pathways)
        - "mlp_wsi":  no specific documentation, MLP for wsi
        - (comparison) ["transmil_wsi", "transmil_wsi_pathways"]: Attention MIL Implementation for unimodal (wsi) and multimodal (omics + wsi)
        - (comparison) "coattn":  MCAT architecture but with pathways instead of the 6 gene families
        - (comparison) "coattn_motcat":  model not present in the code
        (survpath)
        - "survpath": 
    '''
    
    print('\nInit Model...', end=' ')  
    
    # checks type of path for omics (xena, hallmarks, combine, multi) given as args
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        # composition_df contains the pathways compositions
        model = MaskedOmics(**model_dict)
        # aggregate the genes into pathways and then pass through a fully connected layer to get the predictions (existing study)
        # For comparison with MLP transcriptomics only

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)
        # multilayer perceptron MLP to handle tabular omics data (existing study)
        # 3.1 b of paper?

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
        }
        model = SNNOmics(**model_dict)
        # Implement a genomics self normalizing network to handle tabular omics data (existing study)
        # For Comaparison on transcriptomics only

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)
        # Implement Attention MIL (multiple instance learning) for the unimodal (WSI only) and multimodal setting (pathways + WSI), (existing study)
        # Used to compare with ABMIL, late fusion methods

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)
        # Implements DeepMISL for unimodal (WSI only) and multimodal (WSI + pathways)  (existing study)
        # Used to compare with ASMIL, late fusion methods

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)
        # no specific documentation, MLP for wsi

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)
        # Attention MIL Implementation for unimodal (wsi) and multimodal (omics + wsi)
        # For Comparison on unimodal Histology/multimodal

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)
        # MCAT architecture but with pathways instead of the 6 gene families (existing study)
        # For comparison with early fusion methods

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)
        # not present in the code
        # For comparison with early fusion methods

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)  # for ablation studies
        else:
            model = SurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data.
    Return structured array of censorship val and labels (event times) for both train and val ds.
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(), #get censorship vars for train and val, concatenate them as rows 
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)  # axis 0 - rows

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),  # same for labels
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times) # Surv: library of sklearn to do survival analysis; Create structured array
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    Add wsi and omics data to devide. Unpack data in 'data' (from loader): y_disc, event_time, censor, clinical_data_list, mask (from data[6]) and add it to devide
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:   # unimodal, genomics
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:  # unimodal, wsi / multimodal. Both comparison
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:  # just comparison 
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath"]:  ## our model
        
        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[6][0,0] == 1:     # what is here?
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask


def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    Extracts data from Loader calling specific fuction, calls model on the data. Returns output of the model and other info retrieved from the loader.
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    # extract data indicated below from the loader, and add it to the device - returns two times mask (that may be None)
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat"]:   # just comparison
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality == 'survpath':   ## our model

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)  # in sequence takes: data_omics, y_disc, event_time, censor, clinical_data_list, mask
        input_args["return_attn"] = False
        out = model(**input_args)   # gives both wsi and omics data to the model - forward function is called
        
    else:   # can be ignored
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)   # Returns a new tensor with a dimension of size one inserted at the specified position (row! everything is in same row). The returned tensor shares the same underlying data with this tensor
    return out, y_disc, event_time, censor, clinical_data_list 


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient.
    We build a classifier such that each output logit predicted by the network correspond to a time interval.
    we define the discrete hazard function fhazard(yj|x ̄Att) = S(yˆj) where S is the sigmoid activation: fhazard(yj| ̄xAtt) represents the probability that the patient dies during time interval (tj−1, tj).
    we define the discrete survival function fsurv(yj|x ̄Att) =  productur from k=1 to h of (1 − fhazard(yk|x ̄Att)) that represents the probability that the patient survives up to time interval (tj−1, tj).
    These enable us to define the negative log-likelihood (NLL) survival loss, which generalizes NLL to data with censorship.
    by taking the negative of the sum of all logits, we can define a patient-level risk used to identify different risk groups and stratify patients.

    Args: 
        - h : torch.Tensor (output of the model for specific epoch and data)
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values. Add new risk scores, new censorship info, new event times info, clinical data info collected by the batch of specific epoch
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training. 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float   (loss of one epoch. loss given by sum of all loss of single batches, divided by lenght of ds)
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # Set the module in training mode.

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()  # resets the gradient. clearing them ensures that each optimization step is based only on the current batch of data, preventing incorrect updates.

        # h is the output of the model on the data indicated
        h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        
        loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)  # NLLSurvLoss
        loss_value = loss.item()  # Returns the value of this tensor as a standard Python number
        loss = loss / y_disc.shape[0] # divide by num of samples
        
        risk, _ = _calculate_risk(h)  # get the risk for the patients (calculate survival), and detach from gpu the survival values

        # add new info collected during batch to general array of epoch
        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

        # sum all loss
        total_loss += loss_value 

        # optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    # when the batches of epoch are over:
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0) 
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    # calculate c-index: concordance_index_censored of the epoch
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # from sklearn: The concordance index is defined as the proportion of all comparable pairs in which the predictions and outcomes are concordant.
    # Two samples are comparable if (i) both of them experienced an event (at different times), or (ii) the one with a shorter observed survival time experienced an event, 
    # in which case the event-free subject “outlived” the other. 
    # A pair is not comparable if they experienced events at the same time.
    # Concordance intuitively means that two samples were ordered correctly by the model. 
    # More specifically, two samples are concordant, if the one with a higher estimated risk score has a shorter actual survival time. 
    # When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count of concordant pairs.

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics (c-index, c_index_ipcw, BS, IBS, iauc)
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model. Return results of run, total loss, and metrics
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:

            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["coattn", "coattn_motcat"]:  # just comparison
                h = model(
                    x_path=data_WSI, 
                    x_omic1=data_omics[0], 
                    x_omic2=data_omics[1], 
                    x_omic3=data_omics[2], 
                    x_omic4=data_omics[3], 
                    x_omic5=data_omics[4], 
                    x_omic6=data_omics[5]
                )  

            elif modality == "survpath":  ## our model
                # get data from loaders
                input_args = {"x_path": data_WSI.to(device)}
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                
                h = model(**input_args)  # get model output
                
            else:   ## other comparison
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            risk, risk_by_bin = _calculate_risk(h) # get risk of the patients
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # put important info in new dictionary, divided by case_id (first 12 values of slide_ids)
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    # returns results of validation run, plus total loss and metrics
    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    r'''
    Learning Rate Scheduler: sophisticated mechanism to dynamically adjust this hyperparameter lr as the training progresses
    '''
    # get scheduler and info on epochs from args
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0

    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )

    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Called: _step(cur (num of fold), args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur  (num of fold)
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    # get structured array with info on censorship and event times from both train and val (together)
    all_survival = _extract_survival_metadata(train_loader, val_loader)  
    
    for epoch in range(args.max_epochs):
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # save the trained model
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    # validation run, no trainig active
    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    
    print('Final Val c-index: {:.4f}'.format(val_cindex)) # print c-index of validation run
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int       (num of fold)
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize (save splits plus useful print)
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function, calls for NLLSurvLoss
    loss_fn = _init_loss_function(args)

    #----> init model, get model needed (if for survpath or comparison studies, according to modality)
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders for train and val
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do training of the model and validation of the trained model. Returns the results of the validation run and the metrics
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
