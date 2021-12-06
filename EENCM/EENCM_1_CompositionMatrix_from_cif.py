#### Original R script is EENCM_1_CompositionMatrix_from_cif.R
#### Converted to Python by Donggeon Lee in Kyungpook National University

import pandas as pd
import numpy as np
import os, re


def chemical_formula_search_and_split(cif):
    f = open(cif, 'r')
    lines = f.readlines()
    for line in lines:
        if line.find("_chemical_formula_sum") == 0:
            extracted = line.split("'")[1].split(" ")
            element_set = dict()
            for element in extracted:
                el_name = re.sub(r'[0-9]+', "", element)
                el_num = int(re.findall(r'\d+', element)[0])
                element_set[el_name] = el_num
            return element_set

if __name__=="__main__":
    #### Load cif file
    dir_cif = "./Data_cif_sample/"
    vList_cif = os.listdir(dir_cif)
    vList_name = []
    for mp in vList_cif:
        vList_name += [mp[:-4]]

    PT = pd.read_csv("periodic_table_by_R_package.csv")[1:]
    vElement = PT["symb"]

    #### Make matrix 
    df_MP = pd.DataFrame(np.zeros((len(vList_cif), len(vElement)+1), dtype=int))
    df_MP.columns = ["name"] + list(vElement.values)
    df_MP["name"] = vList_name

    for cif in range(0, len(df_MP)):
        M_name = df_MP["name"][cif]
        cif_file = dir_cif+"%s.cif"%(M_name)
        elements_set = chemical_formula_search_and_split(cif_file)
        elements_set = list(elements_set.items())
        for i in range(len(elements_set)):
            df_MP.at[cif, elements_set[i][0]] = elements_set[i][1]

    #### Normalization: TF
    df_TF = df_MP.iloc[:, 1:]
    cRowsum = df_TF.sum(axis=1)
    for r in range(len(df_TF)):
        df_TF.loc[r] = df_TF.loc[r]/cRowsum[r]
    df_TF_vColsum = (df_TF != 0).sum(axis=0)
    df_CM_TF = df_TF.T[df_TF_vColsum != 0].T

    #### Normalization: iDF
    df_CM_IDF = pd.DataFrame(np.zeros((df_CM_TF.shape[0], df_CM_TF.shape[1]), dtype=int))
    df_CM_IDF.columns = df_CM_TF.columns
    vColsum = (df_CM_TF != 0).sum(axis=0)
    vTotalRow = len(df_CM_TF)
    for i in range(len(df_CM_TF.columns)):
        df_CM_IDF.iloc[:, i] = np.log(vTotalRow/vColsum[i])

    #### Normalization: TF*iDF
    df_CM_TFIDF = df_CM_TF*df_CM_IDF
    print(df_CM_TFIDF.head())
    print(df_CM_TFIDF.shape)

    #### Save: csv type
    df_CM_TFIDF.to_csv("1_CompositionMatrix_TFIDF.csv", index=False)

    #### Save: element name
    vElement = df_CM_TFIDF.columns
    np.savetxt("1_CompositionMatrix_element.txt", vElement, delimiter=",", fmt="%s")