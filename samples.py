import df_utils as dfu
import numpy as np
import pandas as pd
from copy import deepcopy
import ROOT as r
import os
from root_numpy import tree2array

# -- This removes a very noisy numpy warning  
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')

@np.vectorize
def split_columns(arr, reti = 0, max_multiplicity = 5, dummy_val = -99):
  ''' This function can be used to split lists into columns, 
      padding non-existent values with a dummy value '''
  arr = np.pad(arr, (0, abs(max_multiplicity-len(arr))))
  arr = np.where(arr != 0, arr, dummy_val)
  return arr[reti]

class smp2df:
  samples = []
  def __init__(self, branches, friends, name = "df"):
    # Main dataframe to be returned by the class
    self.name = name
    self.friends = friends
    self.branches = branches
    self.df = pd.DataFrame()
    return
  
  def process_dataframe(self, df):   #In fact now we don't need this
    ''' This function is very specific, but it's the easiest thing to do if we want
    to deal with jagged arrays. It is very unefficient, but works :). '''
      
    # Declare some variables
    nJets = 5 # Consider up to nJets
    for var in ["pt","eta", "phi", "mass","btagDeepFlavB"]:
      for ijet in range(nJets):
        newjet_arr = split_columns(df["JetSel_Recl_%s"%var].to_numpy(), 
                                   reti = ijet, 
                                   max_multiplicity = nJets)
        col = pd.DataFrame(newjet_arr)
        df["jet%d_%s"%(ijet+1, var)] = col
      df = df.drop("JetSel_Recl_%s"%var, axis = 1)
    
    nLep = 3  # Consider up to nLep leptons

    for var in ["pt", "eta", "phi", "mass"]:
      for ilep in range(nLep):
        newjet_arr = split_columns(df["LepGood_%s"%var].to_numpy(), 
                                   reti = ilep, 
                                   max_multiplicity = nLep)
        col = pd.DataFrame(newjet_arr)
        df["lep%d_%s"%(ilep+1, var)] = col
      df = df.drop("LepGood_%s"%var, axis = 1)
    return df
  
  def load_data(self, path, selection="",process="", object_selection={},start = 0, stop = 10):   #stop:numero de eventos a considerar (1000 bn en local, a mas cluster)
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
      
    # -- Load the main tree and add friends
    tfile = r.TFile.Open(os.path.join(path, process + ".root"))    
    ttree = tfile.Get("Friends")
    #for ftree in self.friends:
    #  ttree.AddFriend("Friends", os.path.join(path, ftree, process + "_Friend.root"))
    
    # Get the information as a numpy array and convert into dataframe
    arr = tree2array(ttree,selection=selection,object_selection=object_selection, branches = self.branches, start = start, stop = stop) #Check no 0 jets
    df = pd.DataFrame(arr, columns = self.branches)
    
    # Now process the dataformat a little bit so we have equally dimensioned arrays...
   # df = self.process_dataframe(df)
    
    # Now combine in the class' dataframe 
    self.df = dfu.combine_dataframes([self.df, deepcopy(df)])
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
 


