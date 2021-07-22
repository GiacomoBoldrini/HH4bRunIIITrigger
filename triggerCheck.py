import ROOT
import numpy
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import datetime


@ROOT.Numba.Declare(["RVec<float>", "RVec<float>", "float", "int"], "bool")
def btagSelector(pt, b, th,c):
    # Uggly but needed for c++ conversion
    if len(pt)  == 0: return False
    bt_pt30 =[]
    for i,j in zip(pt, b):
        if i > 30: bt_pt30.append(j)
        
    count = 0
    for i in bt_pt30:
        if i > th: count += 1
    
    return count > c

@ROOT.Numba.Declare(["RVec<unsigned int>", "RVec<int>"], "bool")
def L1FilterPrescaled(l1bits, l1Prescale):
    # Uggly but needed for c++ conversion
    psL1 = np.zeros(len(l1bits), np.int32)
    i = 0
    while i < len(l1bits):
        psL1[i] = (l1bits[i] * l1Prescale[i])
        i+=1
    
    for j in psL1:
        if j == True: return True
    
    return False

@ROOT.Numba.Declare(["RVec<unsigned int>"], "bool")
def L1Filter(l1bits):
    # Uggly but needed for c++ conversion
    # numba does not know python any() builtin
    for j in l1bits:
        if j == True: return True
    
    return False

class newTrigger:
    
    def __init__(self, df):
        # df is a RootDataFrames
        self.df = df
        
        # List of list with two entries, the first element is the base cut for the 
        # Filter method while the second entry is the parameter for a trigger filter
        self.Selections = [
            ["Numba::L1FilterPrescaled(trigger_l1_pass, trigger_l1_prescale)", []],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [0, 0, 60]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [1, 1, 40]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [2, 2, 30]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [0, 0, 70]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [1, 1, 50]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [2, 2, 45]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [3, 3, 25]],
            ["Numba::btagSelector(jet_hlt_pt, jet_hlt_pnet_probb, {}, {})" , [0.5, 0]],
            ["Numba::btagSelector(jet_hlt_pt, jet_hlt_pnet_probb, {}, {})" , [0.3, 1]]
        ]
        
        # Stores the count of events at each filter in a sequential manner
        self.bits = []
        
        
    def modifySel(self, idx, new):
        # Allows to change a selection item
        self.Selections[idx] = new
        
    def setDF(self, df):
        # Allows to change the dataset on which the trigger selection is performed
        self.df = df
        
    def runTrigger(self):
        # Runs all the trigger selection with Filter method 
        # Stores the result in the bits
        
        #reset bit count
        self.bits = []
        
        # Value-copy the dataframe. Actually the Filter option does not cancel events 
        # from the datafram but only masks them but i do not know how to cancel previous selections.
        df = self.df
        
        # Running selection for each filter defined in the Selection attribute
        for sel in self.Selections:
            cut  = sel[0].format(*sel[1]) # Complete the cut string with pars
            df = df.Filter(cut)
            c = df.Count() #retrieving the count at this stage
            print(c.GetValue()) # Verbosity
            self.bits.append(c.GetValue())
        
    def passed(self):
        # Return overall number of events accepted by the full trigger
        return self.bits[-1] 
        
    def getBits(self):
        # Get full history of selections
        return self.bits
        
        
if __name__ == "__main__":
    
    # Enable multitheading
    ROOT.ROOT.EnableImplicitMT(6)
    
    #start chronometer 
    begin_time = datetime.datetime.now()
    
    # Retireving all files for signal and background
    f_sig = glob("../roots/ggHH_4b_kl_1.0/*")
    f_bkg  = glob("../roots/Ephemeral8/*.root") + glob("../roots/Ephemeral7/*.root") + glob("../roots/Ephemeral6/*.root") + glob("../roots/Ephemeral5/*.root") + glob("../roots/Ephemeral4/*.root") + glob("../roots/Ephemeral3/*.root") + glob("../roots/Ephemeral2/*.root") + glob("../roots/Ephemeral1/*.root")


    
    # Build root dataframes to analyse data
    t_sig=ROOT.RDataFrame("dnntree/tree", "../roots/ggHH_4b_kl_1.0/*.root")
    t_bkg=ROOT.RDataFrame("dnntree/tree", f_bkg)
    
    # Filter events with negative weight
    t_sig = t_sig.Filter("wgt > 0")
    
    # retireve total number of events
    # for signal this is really the total number of events 
    # for bkg this is the number of events after L1 (ephemeral minimum bias)
    tot_sig = t_sig.Count().GetValue()
    tot_bkg = t_bkg.Count().GetValue()
    
    print("Events count. Sig: {}, Bkg: {}".format(tot_sig, tot_bkg))
    
    #build trigger and compute benchmark signal acceptanxce
    tr = newTrigger(t_sig)
    tr.runTrigger()
    sig_acc = float(tr.passed())/tot_sig
    
    #compute benchmark data  zerobias acceptance
    # L1 selection does nothing on this 
    tr.setDF(t_bkg)
    tr.runTrigger()
    bkg_acc = float(tr.passed())/tot_bkg 
    
    
    print(datetime.datetime.now() - begin_time)
    print("Benchmark performance. Signal: {:.3f}%, Background: {:.3f}%".format(sig_acc*100, bkg_acc*100))