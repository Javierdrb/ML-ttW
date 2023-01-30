import df_utils as dfu
import numpy as np
import pandas as pd
from copy import deepcopy
import ROOT as r
import os
from root_numpy import tree2array

class smp2df:
  samples = []
  def __init__(self, branches, friends, name = "df"):
    # Main dataframe to be returned by the class
    self.name = name
    self.friends = friends
    self.branches = branches
    self.df = pd.DataFrame()
    return

  def load_data(self, path, process, start = 0, stop = 1000):
    '''
    This function returns a pandas dataframe in which each row corresponds to
    an event, and each column is a variable.
    '''
    
    if isinstance(process, list):
      for p in process: 
        self.load_data(path, p, start, stop)   
      return
    
    # Save some info into attributes for book keeping purposes
    self.samples.append([path, process])
      
    # -- Load the main tree 
    tfile = r.TFile.Open(os.path.join(path, process + ".root"))    
    ttree = tfile.Get("Events")

    # -- Now add friends
    for ftree in self.friends:
      ttree.AddFriend("Friends", os.path.join(path, ftree, process + "_Friend.root"))
    arr = tree2array(ttree, branches = self.branches, start = start, stop = stop)

    # Convert into dataframe and concatenate with the main one
    self.df = dfu.combine_dataframes([self.df, deepcopy(pd.DataFrame(arr, columns = self.branches))])
    tfile.Close()
    return
  
  def summary(self):
    print(" >> Summary of dataframe: %s"%self.name)
    print(" - Branches: %s"%self.branches)
    print(" - Friends: %s"%self.friends)
    print(" - Samples considered:")
    for s in self.samples:
      print("\t * %s (stored in %s)"%(s[1], s[0]))
    print(" >> End of summary")
    return

  def label_dataframe(self, label = "is_signal", val = 0):
    '''
    Function used to label dataframes. Required for supervised models.
    '''
    nentries = self.df.shape[0]
    values = np.array( [np.int32(val)]*nentries )
    self.df = dfu.combine_dataframes([self.df, pd.DataFrame(values, columns = [label])], axis = 1)
    return
 


