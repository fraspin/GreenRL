from datetime import datetime as dt
from time import time 
from os import path
import numpy as np
import yaml
import pickle, pickletools
import time
def load_yaml_configuration(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def read_pickle_objects(path, EVAL_SETUP):
    """
    Read pickle files that store results for each algorithm. To check if some are missing
    """
    print('path',path)
    with open(path, 'rb') as mfile:
        stage = 0
        backup = {}
        try:
            backup["results_eval"]            = pickle.load(mfile)
            backup["env"]                     = pickle.load(mfile)
            stage        += 1
            backup["results_eval_random"]     = pickle.load(mfile)
            backup["results_eval_emptier"]    = pickle.load(mfile)
            backup["results_eval_heuristic"]  = pickle.load(mfile)
            stage        += 1
            backup["results_eval_sota"]       = pickle.load(mfile)
            stage        += 1
            backup["results_eval_carbonedge"] = pickle.load(mfile)
            stage        += 1
        except EOFError:
            pass # End of File    
    # print('stage',stage)
    # print('backup',backup)
    # print('EVAL_SETUP',EVAL_SETUP)
    # time.sleep(20)
    final_stage = verify_evaluation_order(backup, stage, EVAL_SETUP)

    status = {"backup_stage": stage, "final_stage": final_stage }
    return backup, status    
    
def verify_evaluation_order(backup, eval_stage, EVAL_SETUP):
    greenRL_notCompleted = eval_stage == 0 or "results_eval" not in backup.keys() or "env" not in backup.keys()
    backEnv_required     = EVAL_SETUP["BASELINES"] + EVAL_SETUP["SOTA"] + EVAL_SETUP["CARBONEDGE"] + EVAL_SETUP["SOLVER"]

    if greenRL_notCompleted and backEnv_required:
        raise ValueError("GreenRL must be evaluated before any other algorithm. It creates the reference evaluation instance.")

    key_order   = ["POLICY", "BASELINES", "SOTA", "CARBONEDGE"]
    key_suffix  = ["", "_random", "_sota", "_carbonedge"]
    eval_values = [EVAL_SETUP[k] for k in key_order]
    final_stage = 0
    for stage_idx in range(len(eval_values)):
        if eval_values[stage_idx] == 1:
            final_stage += 1
        else:
            backup_key       = f"results_eval{key_suffix[stage_idx]}"
            exists_in_pickle = backup_key in backup.keys()
            if exists_in_pickle: 
                final_stage += 1 # it is not evaluated now but it already exists in the pickle
            elif sum(eval_values[stage_idx:]) > 0: # not data but later stages are supposed to run -- problem           
                raise ValueError(f"Stage {stage_idx} ({key_order[stage_idx]}) must be evaluated before {key_order[stage_idx + 1:]}\n However, no {backup_key} key was found in the backup file from the pickle.")
            else:
                break # no data and no later stages are supposed to run
    return final_stage 

                       
def append_( myList, newValue):
    ''' append method that covers the case in which list is empty '''
    if myList ==  []:
        myList = [newValue]
    else:
        myList.append(newValue)
    return myList     
        
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def folderPathGenerator(mother_folder=".", folder_name="saved_model", ):  # change if you want
    '''
        mother_folder -> The directory where you want to create the folder
        folder_name -> name of the new folder to store all data
    '''

    str_time  = dt.now().strftime("%Y-%m-%d_%H-%M")
    path_name = path.join(mother_folder, folder_name + str_time)

    ii_version = 0
    while path.isdir(path_name):
        ii_version += 1
        path_name = path.join(mother_folder, folder_name + str_time + '_v' + str(ii_version))

    print('Created folder for storage: ', path,'\n')
    return path_name
    
def yesNoQuestion(text):
    value = input(text)
    while not (value in ['y','n']):
        value = input("!!! You have to input either 'y' or 'n', please !!!\n"+text)
    return value == 'y'

def checkFile_and_rename(filePath):
    if path.isfile(filePath):
        numb = 1
        while True:
            newPath = "{0}_{2}{1}".format(*path.splitext(filePath) + (numb,))
            if path.isfile(newPath):
                numb += 1
            else:
                return newPath
    return filePath


class PrintSwitch():
    def __init__(self, doPrint):
        self.print = doPrint
        if doPrint:
            self.print_ = lambda x: print(x)
        else:
            self.print_ = lambda x: False

    def __call__(self, text):
        self.print_(text)
        
class TimeCounter():

    def __init__(self, doPrint):
        self.startTimeGlobal = time()
        self.startTimes = []
        self.print = PrintSwitch(doPrint)
        
    def setTimeOn(self):
      self.startTimes.append(time())
      return len(self.startTimes)
  
    def getTimeOff(self, text = "", pos=-1): #, doPrint = True
        executionTime = (time() - self.startTimes[pos])        
        self.print(text + 'Exec. time (s): %.2f' % executionTime)
        return executionTime
        
    def restartGlobalTime(self):
        self.startTimeGlobal = time()
        
    def getTotalTime(self):
        executionTime = (time() - self.startTimeGlobal)        
        self.print('Total Execution time (s) for whole program: %.2f' % executionTime,'\n')
        return executionTime

    def offAndOn(self, text = "", pos=-1): #, doPrint = True
      exTime = self.getTimeOff(text, pos)
      self.setTimeOn()
      return exTime, len(self.startTimes)
  
