import ROOT
import numpy
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import datetime
from itertools import product
from tqdm import tqdm
import multiprocessing 
import time

@ROOT.Numba.Declare(["RVec<float>", "RVec<float>", "RVec<float>", "RVec<float>", "RVec<float>", "float", "int"], "bool")
def btagSelectorDenom(pt, bb, bc, busd, bg, th,c):
    # Uggly but needed for c++ conversion
    if len(pt)  == 0: return False
    bt_pt30 = []
    for i,j,k,l,m in zip(pt, bb,  bc, busd, bg):
        if i > 30: bt_pt30.append(j / (j +  k + l + m))
        
    count = 0
    for i in bt_pt30:
        if i > th: count += 1
    
    return count > c


@ROOT.Numba.Declare(["RVec<float>", "RVec<float>", "RVec<float>", "RVec<float>", "RVec<float>", "float", "int"], "bool")
def btagSelectorDenom_2(pt, bb, bc, busd, bg, th,c):
    # Uggly but needed for c++ conversion
    if len(pt)  == 0: return False
    bt_pt30 = []
    for i,j,k,l,m in zip(pt, bb,  bc, busd, bg):
        if i > 30: bt_pt30.append(j / (j +  k + l + m))
    
    bt_pt30 = np.array(bt_pt30)
    bt_pt30[::-1].sort()
    
    if bt_pt30[c] > th: return True 
    else: return False

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

@ROOT.Numba.Declare(["RVec<unsigned int>"], "bool")
def L1FilterAdHoc(l1bits):
    
    if l1bits[7]: return True 
    if l1bits[0]: return True 
    if l1bits[5]: return True
    
    return False
    
    
@ROOT.Numba.Declare(["RVec<float>", "RVec<float>"], "bool")
def METL1(l1jpt, l1jeta):
    
    if len(l1jpt) == 0: return False
    ht = 0
    
    for i in range(len(l1jpt)):
        if l1jpt[i] > 30 and abs(l1jeta[i]) < 2.5: ht += l1jpt[i]
        
    if ht > 240: return  True 
    else: return  False 

@ROOT.Numba.Declare(["RVec<float>", "RVec<float>", "RVec<float>"], "bool")
def MuonL1(mupt, mueta, muqual):
    
    if len(mupt) == 0: return False
    mupt_ = 0
    for i  in range(len(mupt)):
        if abs(mueta[i]) < 2.4 and muqual[i] >= 12: 
            if mupt[i] > mupt_: mupt_ = mupt[i]
    
    if mupt_ > 6: return  True 
    else: return  False
    

class newTrigger(multiprocessing.Process):
    
    def __init__(self, df):
        
        multiprocessing.Process.__init__(self)
        
        # df is a RootDataFrames
        self.df = df
        
        # List of list with two entries, the first element is the base cut for the 
        # Filter method while the second entry is the parameter for a trigger filter
        self.Selections = [
            ["Numba::L1FilterAdHoc(trigger_l1_pass) || (Numba::METL1(trigger_l1_jet_pt,trigger_l1_jet_eta) && Numba::MuonL1(trigger_l1_muon_pt, trigger_l1_muon_eta, trigger_l1_muon_qual))", []],
            #["Numba::L1FilterAdHoc(trigger_l1_pass)", []],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [0, 0, 60]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [1, 1, 40]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [2, 2, 30]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [0, 0, 70]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [1, 1, 50]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [2, 2, 40]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [3, 3, 25]],
            ["Numba::btagSelectorDenom_2(jet_hlt_pt, jet_hlt_pnet_probb, jet_hlt_pnet_probc, jet_hlt_pnet_probuds, jet_hlt_pnet_probg, {}, {})" , [0.5, 0]],
            ["Numba::btagSelectorDenom_2(jet_hlt_pt, jet_hlt_pnet_probb, jet_hlt_pnet_probc, jet_hlt_pnet_probuds, jet_hlt_pnet_probg, {}, {})" , [0.3, 1]]
        ]
        
        # Stores the count of events at each filter in a sequential manner
        # Shared between process in asynchronous way
        self.bits = multiprocessing.Array('i', range(len(self.Selections)))
        
        # Set a dummy id
        self.id_ = 0
        
        
    def modifySel(self, idx, new):
        # Allows to change a selection item
        self.Selections[idx] = new
        
    def setSelection(self, selections):
        self.Selections = selections
        
    def setRange(self, r_):
        # multiprocessing
        self.df = self.df.Range(r_[0], r_[1])
        
    def setDF(self, df):
        # Allows to change the dataset on which the trigger selection is performed
        self.df = df
        
    def run(self):
        # Runs all the trigger selection with Filter method 
        # Stores the result in the bits
        
        # Value-copy the dataframe. Actually the Filter option does not cancel events 
        # from the datafram but only masks them but i do not know how to cancel previous selections.
        df = self.df
        
        # Running selection for each filter defined in the Selection attribute
        for i, sel in enumerate(self.Selections):
            cut  = sel[0].format(*sel[1]) # Complete the cut string with pars
            df = df.Filter(cut)
            c = df.Count().GetValue() #retrieving the count at this stage
            # print(c.GetValue()) # Verbosity
            #if hasattr(self, "id_"):
            #    print(c, self.id_)
            #else:
            #    print(c)
            self.bits[i] = c
        
    def passed(self):
        # Return overall number of events accepted by the full trigger
        return self.bits[-1] 
        
    def getBits(self):
        # Get full history of selections
        return self.bits
    
    
def makeRange(tot, nproc):
    step_size = tot // nproc
    range_ = np.arange(0, tot, step_size)
    # last step would be [tot - step_size, 0], tells root to take every entry until eof
    range_ = np.append(range_, 0)
    range_ = [int(i) for i in range_]
    return range_
        
        
if __name__ == "__main__":
    
    # Enable multitheading
    nproc = 4
    
    # Retireving all files for signal and background
    f_bkg  = glob("../roots/Ephemeral8/*.root") + glob("../roots/Ephemeral7/*.root") + glob("../roots/Ephemeral6/*.root") + glob("../roots/Ephemeral5/*.root") + glob("../roots/Ephemeral4/*.root") + glob("../roots/Ephemeral3/*.root") + glob("../roots/Ephemeral2/*.root") + glob("../roots/Ephemeral1/*.root")
    
    
    # Build root dataframes to analyse data
    t_sig=ROOT.RDataFrame("dnntree/tree", "../roots/ggHH_4b_kl_1.0/*.root")
    t_bkg=ROOT.RDataFrame("dnntree/tree", f_bkg)
    
    # Filter events with negative weight
    # t_sig = t_sig.Filter("wgt > 0")
    
    # retireve total number of events
    # for signal this is really the total number of events 
    # for bkg this is the number of events after L1 (ephemeral minimum bias)
    tot_sig = t_sig.Count().GetValue()
    tot_bkg = t_bkg.Count().GetValue()
    
    print("Events count. Sig: {}, Bkg: {}".format(tot_sig, tot_bkg))
    
    # making ranges for multiprocess
    range_signal = makeRange(tot_sig, nproc)
    range_bkg = makeRange(tot_bkg, nproc)
    
    #start chronometer 
    begin_time = datetime.datetime.now()
    
    begin_time = datetime.datetime.now()
    
    # Now multiprocessing
    the_trs = []

    for i in range(len(range_signal)-1):
        t_ = newTrigger(t_sig)
        t_.setRange([range_signal[i], range_signal[i+1]])
        t_.id_ = i
        the_trs.append(t_)
        the_trs[i].start()
        
    jobs = list(the_trs)
    
    # wait until jobs finish
    while len(jobs) > 0:
        jobs = [job for job in jobs if job.is_alive()]
        time.sleep(1)

    print('*** All jobs finished ***')
    print("Parallel time: {}".format(datetime.datetime.now() - begin_time))
        
    # collect results
    bench_sig_acc =  0
    for i in range(len(the_trs)):
        bench_sig_acc += the_trs[i].passed()
        
    bench_sig_acc = float(bench_sig_acc)/tot_sig
    
    the_trs = []

    for i in range(len(range_signal)-1):
        t_ = newTrigger(t_bkg)
        t_.modifySel(0, ["Numba::L1FilterAdHoc(trigger_l1_pass)", []])
        t_.setRange([range_bkg[i], range_bkg[i+1]])
        t_.id_ = i
        the_trs.append(t_)
        the_trs[i].start()
        
    jobs = list(the_trs)
        
    # wait until jobs finish
    while len(jobs) > 0:
        jobs = [job for job in jobs if job.is_alive()]
        time.sleep(1)

    print('*** All jobs finished ***')
        
    # collect results
    bench_bkg_acc =  0
    for trig in the_trs:
        bench_bkg_acc += trig.passed()
        
    bench_bkg_acc = float(bench_bkg_acc)/tot_bkg 
        
        
    print("Parallel time: {}".format(datetime.datetime.now() - begin_time))
    print("Benchmark performance. Signal: {:.3f}%, Background: {:.3f}%".format(bench_sig_acc*100, bench_bkg_acc*100))
    
    
    #  Now moving to the scan
    
    # Requires too much time
    # thresholds = list(product(
    #     *[
    #         np.arange(40,80,15), # Leading calo pt th
    #         np.arange(30,50,10), # 2nd Leading calo pt th
    #         np.arange(20,40,2), # 3rd Leading calo pt th
    #         np.arange(65,95,15), # Leading pf pt th
    #         np.arange(40,80,10), # 2nd Leading pf pt th
    #         np.arange(30,60,5), # 3rd Leading pf pt th
    #         np.arange(20,30,2), # 3rd Leading pf pt th
    #         np.arange(0.3,0.7,0.1), # Leading b-tag
    #         np.arange(0.1,0.5,0.1), # 2nd Leading b-tag
    #     ]
    # ))
    
    
    """
    
    # simple try with onliu btag filters 
    thresholds = list(product(
        *[
            np.array([60]), # Leading calo pt th
            np.array([40]), # 2nd Leading calo pt th
            np.array([30]), # 3rd Leading calo pt th
            np.array([70]), # Leading pf pt th
            np.array([50]), # 2nd Leading pf pt th
            np.array([30, 40, 50]), # 3rd Leading pf pt th
            np.array([25]), # 3rd Leading pf pt th
            np.arange(0.4,0.6,0.05), # Leading b-tag
            np.arange(0.2,0.4,0.05), # 2nd Leading b-tag
        ]
    ))
    
    Selections = [
            ["Numba::L1FilterAdHoc(trigger_l1_pass)", []],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [0, 0, 60]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [1, 1, 40]],
            ["calojet_hlt_pt.size() > {} && calojet_hlt_pt[{}] > {}" , [2, 2, 30]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [0, 0, 70]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [1, 1, 50]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [2, 2, 40]],
            ["jet_hlt_pt.size() > {} && jet_hlt_pt[{}] > {}" , [3, 3, 25]],
            ["Numba::btagSelector(jet_hlt_pt, jet_hlt_pnet_probb, {}, {})" , [0.5, 0]],
            ["Numba::btagSelector(jet_hlt_pt, jet_hlt_pnet_probb, {}, {})" , [0.3, 1]]
        ]
    
    f = open("scan.txt", "w")
    f.write("{} {}\n".format(bench_sig_acc, bench_bkg_acc))
    
    for idx in tqdm(range(len(thresholds))):
        combo = thresholds[idx]
        
        # modify selections object
        for i_ in range(len(combo)):
            s = Selections[i_+1]
            #first seven for pt 
            if i_ < 7:
                Selections[i_+1][1][2] =  combo[i_]
            else:
                Selections[i_+1][1][0] =  round(combo[i_],2)
                
        # Now selection is modified we can  import it into the trigger
        
        print(Selections)
        
        # Signal
        the_trs = []

        # Ddefine and setup the triggers
        for i in range(len(range_signal)-1):
            t_ = newTrigger(t_sig)
            t_.setRange([range_signal[i], range_signal[i+1]])
            t_.id_ = i
            t_.setSelection(Selections)
            the_trs.append(t_)
            the_trs[i].start()
            
        jobs = list(the_trs)
        
        # wait until jobs finish
        while len(jobs) > 0:
            jobs = [job for job in jobs if job.is_alive()]
            time.sleep(1)

        print('*** All jobs finished ***')
        print("Parallel time: {}".format(datetime.datetime.now() - begin_time))
            
        # collect results
        sig_acc =  0
        for i in range(len(the_trs)):
            sig_acc += the_trs[i].passed()
        
        # compute acceptance
        sig_acc = float(sig_acc)/tot_sig
        
        # Background
        the_trs = []
        
        for i in range(len(range_signal)-1):
            t_ = newTrigger(t_bkg)
            t_.setRange([range_bkg[i], range_bkg[i+1]])
            t_.id_ = i
            t_.setSelection(Selections)
            the_trs.append(t_)
            the_trs[i].start()
            
        jobs = list(the_trs)
            
        # wait until jobs finish
        while len(jobs) > 0:
            jobs = [job for job in jobs if job.is_alive()]
            time.sleep(1)

        print('*** All jobs finished ***')
            
        # collect results
        bkg_acc =  0
        for trig in the_trs:
            bkg_acc += trig.passed()
            
        bkg_acc = float(bkg_acc)/tot_bkg 
        
        tow = " ".join([str(cut) for cut in combo])
        print(tow)
        f.write("{} {} {}\n".format(sig_acc, bkg_acc, tow))
        
        
    f.close()
    
    """