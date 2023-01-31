# -- Import libraries 
import pandas as pd
import numpy as np
import ROOT as r


def combine_dataframes(dfs, axis = 0):
    '''
    This function combines multiple dataframes:
      axis=0 -- concatenate rows (i.e. add more events)
      axis=1 -- concatenate columns (i.e. add more features)
    '''
    ignore_index = True if axis == 0 else False
    super_df = pd.concat(dfs, axis, ignore_index = ignore_index) 
    return super_df

   
