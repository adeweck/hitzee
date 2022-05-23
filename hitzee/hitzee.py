import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt

#import loader.read_plates_data as read_plates_data
#import loader.read_plates_annot as read_plates_annot
from hitzee.loader import read_plates_data, read_plates_annot


def z_score_standardization(series):
    return (series - series.mean()) / series.std()



# utils
def removeNAs(df, na_convert_dict = {'object':'NA', 'float64':0.0, 'int64':0}):
    
    # remove dtype specific NA fill
    col_dtypes_dict = {dtype: [] for dtype in na_convert_dict.keys()}
    for col in df.columns.to_list():
        col_dtypes_dict[str(df[col].dtypes)] += [col]
    for dtype in na_convert_dict.keys():
        df[col_dtypes_dict[dtype]] = df[col_dtypes_dict[dtype]].fillna(na_convert_dict[dtype])
    
    return df



def rle_norm(plates_df, output='zscore'):
    
    # RLE normalization
    pseudoref = pd.Series(np.exp(np.log(plates_df).median(axis=1)))
    scaling_factors = plates_df.divide(pseudoref, axis=0).median(axis=0)
    norm_df = plates_df.divide(scaling_factors)

    logfc_df = pd.DataFrame(np.log(norm_df)).subtract(np.log(pseudoref), axis=0)

    zscore_df = logfc_df.copy()
    for col in zscore_df.columns:
        zscore_df[col] = -z_score_standardization(zscore_df[col])
        
    if output == 'logfc':
        output_df = logfc_df
    else:
        output_df = zscore_df
        
    return output_df



def rle_plus_norm(plates_df, output='zscore', topN=5):

    # RLE normalization (remove plate effect)
    pseudoref = pd.Series(np.exp(np.log(plates_df).median(axis=1)))
    # pseudoref = pd.Series(np.exp(np.log(plates_df).mean(axis=1)))
    scaling_factors = plates_df.divide(pseudoref, axis=0).median(axis=0)
    norm_df0 = plates_df.divide(scaling_factors)
    
    # Local transposed RLE normalization (remove positional effect)
    plates_df0 = norm_df0.transpose()

    tmp_df0 = np.log(plates_df0)
    tmp_df_corr = tmp_df0.corr(method='spearman').clip(lower=0)
    df_thrshld = tmp_df_corr.copy()
    for col in tmp_df_corr:
        df_thrshld[col] = list(tmp_df_corr[col].sort_values(ignore_index=True, ascending=False))
    sr_thrshld = df_thrshld.iloc[topN]
    tmp_df_corr = tmp_df_corr.where(tmp_df_corr >= sr_thrshld).fillna(0)
    tmp_df_corr[tmp_df_corr > 0] = 1

    tmp_df = pd.DataFrame(np.matmul(tmp_df0, np.asarray(tmp_df_corr)))
    tmp_df.columns = plates_df0.columns
    tmp_df = (tmp_df - tmp_df0) / (tmp_df_corr.sum()-1)
    #tmp_df = tmp_df / (tmp_df_corr.sum()-1)
    tmp_df = np.exp(tmp_df)

    scaling_factors = plates_df0.divide(tmp_df, axis=0).median(axis=0)
    pseudoref_df = tmp_df.multiply(scaling_factors)

    logfc_df = pd.DataFrame(np.log(plates_df0)).subtract(np.log(pseudoref_df), axis=0).transpose()

    zscore_df = logfc_df.copy()
    for col in zscore_df.columns:
        zscore_df[col] = -z_score_standardization(zscore_df[col])
        
    if output == 'logfc':
        output_df = logfc_df
    else:
        output_df = zscore_df
        
    return output_df



def get_hit_scores(plates_df, annot_df=None):
    
    zscore1 = rle_norm(plates_df)
    zscore2 = rle_plus_norm(plates_df)
    
    tmp_df = plates_df.copy()
    tmp_df['position'] = tmp_df.index
    tmp_df[['row','col']] = tmp_df['position'].str.split('-', 1, expand=True)
    tmp_df = tmp_df.melt(id_vars=['position','row','col'], var_name='plate', value_name='raw')
    tmp_df['zscore_rle'] = zscore1.melt().value
    tmp_df['zscore_rle_plus'] = zscore2.melt().value
    
    if annot_df is not None:
        #annot_df['position'] = annot_df.index.copy()
        annot_df = annot_df.reset_index().rename(columns={'index':'position'})
        annot_df = annot_df.melt(id_vars=['position'], var_name='plate', value_name='cmpd')

        tmp_df = tmp_df.merge(annot_df, how='left', on=['position','plate'])
    
    return tmp_df


def get_batch_separation(plates_df, plates_annot_df=None, dendo_cut_sd=5, linkeage_method='average'):
    
    linked = linkage(plates_df.transpose(), linkeage_method)#, optimal_ordering=True)
    heights = pd.DataFrame(linked, columns=['a','b','c','d'])['c']
    min_height = min(heights)
    mean_height = np.mean(heights - min(heights))
    cut_height = min_height + dendo_cut_sd*mean_height
    cluster_tag = cut_tree(linked, height = cut_height).flatten().tolist()

    labelList = plates_df.columns
    
    plate_group = {}
    for k in range(len(cluster_tag)):
        if cluster_tag[k] in plate_group:
            plate_group[cluster_tag[k]].append(labelList[k])
        else:
            plate_group[cluster_tag[k]] = [labelList[k]]

    plate_batch_df = {}
    plate_annot_df = {}
    outlier_plates = []
    for k in plate_group.keys():
        if len(plate_group[k]) <= 5:
            outlier_plates.extend(plate_group[k])
        else:
            plate_batch_df[k] = plates_df[plate_group[k]]
            if plates_annot_df is not None:
                plate_annot_df[k] = plates_annot_df[plate_group[k]]
                
    if len(outlier_plates) > 0:
        print('Outlier plates')
        print(outlier_plates)
            
    return plate_batch_df, plate_annot_df



def hitzee_norm(plates_df, plates_annot_df=None):
    
    plates_batch_df, plates_batch_annot_df = get_batch_separation(plates_df, plates_annot_df)
    
    hits_df = pd.DataFrame()
    for k in plates_batch_df.keys():
        if plates_annot_df is not None:
            tmp_df = get_hit_scores(plates_batch_df[k], annot_df=plates_batch_annot_df[k])
        else:
            tmp_df = get_hit_scores(plates_batch_df[k])
        tmp_df['batch'] = k
        hits_df = pd.concat([hits_df, tmp_df])
        
    return hits_df



def hitzee(input_dir, annot_dir=None, outfile=None):
    """
    Take all csv files from input_dir and txt files from annot_dir (if available) and \
    calculates position based normalization for plate-based read-outs.
    Batch clustering is performed before normalization.
    """
    
    plates_df = read_plates_data(input_dir)
    
    if annot_dir is not None:
        plates_annot_df = read_plates_annot(annot_dir, rowcol=plates_df.index)
        hits_df = hitzee_norm(plates_df, plates_annot_df)
    else:
        hits_df = hitzee_norm(plates_df)
        
    if outfile is not None:
        hits_df.to_csv(outfile)
        
    return hits_df