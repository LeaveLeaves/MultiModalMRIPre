import pandas as pd
import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import intersect_masks
from pymatch.Matcher import Matcher
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

class preprocess():
    '''A general class to read in, prepross and mask MRI.
    
    Parameters
    ----------
    ND_path: file path
        The neurological disorder(all kinds of neurological disorder, Major depression(MD), bipolar disorder(BD), 
        Multuple Sclerosis or Mild cognitive impairment(MCI)) MRI file path.
    
    HC_path: file path
        The healthy control(HC) MRI file path.
    
    pheno_path: file path
        MRI file path of one group(all kinds of neurological disorder, MD, MS or MCI).
    
    modal_path: file path
        The T1_brain_MNI MRI file path.
    
    '''
    
    def _init_(self, ND_path, HC_path, pheno_path, modal_path):
        self.ND_path = ND_path
        self.HC_path = HC_path
        self.pheno_path = pheno_path
        self.modal_path = modal_path
    
    #read in MRI data, match them to modality and phenotype information
    def read_data(self):
        #read in ND, HC file path
        ND = pd.read_csv(self.ND_path, dtype = str, names = ['eid'], dtype = str)
        HC = pd.read_csv(self.HC_path, dtype = str, names = ['eid'], dtype = str)
        #read in phenotype information
        df = pd.read_csv(self.pheno_path, dtype = str)
        #read in modal file path
        modal = pd.read_csv(self.modal_path, dtype = str, names = ['file_path'], header = None)
        #extract subject id
        modal['eid'] = modal['file_path'].str[2:9]
        #match phenotype information data to modality and healthy status group
        HC_modal = pd.merge(file_path, HC, on = 'eid')
        HC_modal_pheno = pd.merge(df, HC_modal, on = 'eid')
        ND_modal = pd.merge(file_path, ND, on = 'eid')
        ND_modal_pheno = pd.merge(df, ND_modal, on = 'eid')
        
        return HC_modal_pheno, ND_modal_pheno
           
    def PSM(self, HC_modal_pheno, ND_modal_pheno):
        '''A method class to Propensity score matching HC and ND data.
        
        Parameters
        ----------
        HC_modal_pheno: DataFrame
        Phenotype information DataFrame of HC in one modality
    
        ND_modal_pheno: DataFrame
        Phenotype information DataFrame of ND in one modality
        '''
        #select relevant phenotype: sex, age, ethnicity,smoking status, BMI, label samples
        HC_Match = HC_modal_pheno[["eid", "31-0.0", "34-0.0", "file_path"]]
        HC_Match[["34-0.0"]] = HC_Match [["34-0.0"]].astype(float)
        #HC_Match = HC_Match.fillna(method = 'ffill')
        HC_Match['label'] = 0
        
        ND_Match = ND_modal_pheno[["eid", "31-0.0", "34-0.0", "file_path"]]
        ND_Match[["34-0.0"]] = ND_Match [["34-0.0"]].astype(float)
        #ND_Match = ND_Match.fillna(method = 'ffill')
        ND_Match['label'] = 1
        
        #Caulate propensity score and match them
        match_PSM = Matcher(ND_Match, HC_Match, yvar="label", exclude=['eid', 'file_path'])
        np.random.seed(20200624)
        match_PSM.fit_scores(balance = True, nmodels = 1000)
        match_PSM.match(method = 'min', nmatches = 1, threshold = 0.001)
        
        #get the matched balanced data
        HC_ND_matched = match_PSM.matched_data[['eid', 'file_path', 'label']].sort_values('label', ascending = False)
        HC_ND_matched.reset_index(drop = True, inplace = True)
        
        return HC_ND_matched

    #compute and apply mask to T1 MRI
    def matched_mask(self, HC_ND_matched)    
        nifti_masker = NiftiMasker(mask_strategy = 'template', memory = "nilearn_cache", 
                           memory_level = 1)
        mri = []
        for i in range(0, HC_ND_matched.shape[0]):
            mri.append(nifti_masker.fit_transform(HC_ND_matched['file_path'].loc[i]))
            
        retrun np.reshape(np.array(mri), (-1, mri[0].shape[1]))
       
#do nested cross validation using SVM, Logistic regression and random forest
def NestedCV(self, X, y, cv_fold, base_model, p_grid, i)
    accuracy = []
    f1 = []
    recall = []
    precision = []
    spcificty = []
    tprs = []
    aucs = []
    AUROC = []
    AUPR = []
    
    model = base_model
    outer_cv = StratifiedKFold(n_splits = cv_fold, shuffle = True, random_state = i)
    inner_cv = StratifiedKFold(n_splits = cv_fold, shuffle = True, random_state = i)
    
    #run outter CV to evaluate model
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #find optim paramater setting in the inner cv
        clf = GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "recall")
        clf.fit(x_train, y_train)

        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #calculate metrics
        a = metrics.accuracy_score(y_test, y_pred)
        f = metrics.f1_score(y_test, y_pred)
        r = metrics.recall_score(y_test, y_pred, average='binary')
        p = metrics.precision_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s = tn / (tn+fp)
        accuracy.append(a)
        f1.append(f)
        recall.append(r)
        precision.append(p)
        spcificty.append(s)
        y_score = clf.predict_proba(x_test)[:,1]
        pre, re, thresholds = precision_recall_curve(y_test, y_score)
        AUPR.append(auc(re, pre))
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        AUROC.append(auc(fpr, tpr))
    
    return np.mean(accuracy), np.mean(f1), np.mean(recall), np.mean(precision), np.mean(spcificty), np.mean(AUROC), np.mean(AUPR), np.std(accuracy), np.std(f1), np.std(recall), np.std(precision), np.std(spcificty), np.std(AUROC), np.std(AUPR)



    
