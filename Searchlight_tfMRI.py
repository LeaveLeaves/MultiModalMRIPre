import numpy as np
import pandas as pd
from nilearn.image import new_img_like, load_img, get_data
from nilearn.image import resample_to_img
from sklearn.model_selection import KFold
from nilearn.image import index_img
import nilearn.decoding
from nilearn.input_data import NiftiMasker
import nilearn as nil
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from nilearn.masking import intersect_masks
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score 
from nilearn import plotting
from sklearn.metrics import confusion_matrix
from nilearn.datasets import load_mni152_template
import warnings
from nilearn.datasets import load_mni152_template
import time

class Searchlight_tfMRI:
    def _init_(self, ND_HC_path, HC_path, pheno_path, modal_path):
        self.ND_HC_path = ND_HC_path
        self.HC_path = HC_path
        self.pheno_path = pheno_path
        self.modal_path = modal_path
        
    def prerocess(self):
        filename = pd.read_csv(self.ND_HC_path, dtype = str)
        shape = np.repeat('shapes', 21)
        face = np.repeat('faces', 21)
        label = np.repeat('rest', 332)
        designS = [0, 71, 100, 171, 200]
        designF = [21, 50, 121, 150, 221]
        for i in designS:
            label[i:i+21] = shape
        for j in designF:
            label[j:j+21] = face
        labels = pd.DataFrame(label, columns = ['labels'])
        y = labels['labels']#Restrict to shapes and face
        condition_mask = y.isin(['shap', 'face'])
        y = y[condition_mask]
        label1 = np.repeat('rest', 366)
        
        for i in designS:
            label1[i:i+21] = shape
        for j in designF:
            label1[j:j+21] = face
        labels1 = pd.DataFrame(label1, columns = ['labels'])
        y1 = labels1['labels']
        #Restrict to shapes and face
        condition_mask1 = y1.isin(['shap', 'face'])
        y1 = y1[condition_mask1]
        return condition_mask, condition_mask1, y, y1
    
    def using_multiindex(score_matrix, cols):
        shape = score_matrix.shape
        index = pd.MultiIndex.from_product([range(s)for s in shape], names=cols)
        df = pd.DataFrame({'score_matrix': score_matrix.flatten()}, index=index).reset_index()
        return df
    
    def Searchlight_score(self, n_jobs, n_splits, condition_mask, condition_mask1, y, y1):
        cv = KFold(n_splits = n_splits)
        nifti_masker = NiftiMasker(mask_strategy='template', memory="nilearn_cache",  memory_level=1)
        score = []
        for i in range(0, filename.shape[0]):
            start_time = time.time()
            mask_fmri = nifti_masker.fit(filename['file'].loc[i]).mask_img_
            searchlight = nilearn.decoding.SearchLight(mask_fmri, n_jobs=n_jobs, verbose=1, cv=cv)
            if (nib.load(filename['file'].loc[i]).shape[3] == 366):
                fmri_img = index_img(filename['file'].loc[i], condition_mask1)
                searchlight.fit(fmri_img, y1)
            else:
                fmri_img = index_img(filename['file'].loc[i], condition_mask)
                searchlight.fit(fmri_img, y)
            
        score.append(searchlight.scores_)
        return score
        
def label_condition(con1,con2):
    shape = np.repeat('shapes', 21)
    face = np.repeat('faces', 21)
    label = np.repeat('rest', 332)
    designS = [0, 71, 100, 171, 200]
    designF = [21, 50, 121, 150, 221]
    for i in designS:
        label[i:i+21] = shape
    for j in designF:
        label[j:j+21] = face
    labels = pd.DataFrame(label, columns = ['labels'])
    y = labels['labels']
    #Restrict to shapes and face
    #condition_mask = y.isin(['shap', 'face'])
    #condition_mask = y.isin(['rest', 'face'])
    condition_mask = y.isin([con1, con2])
    y = y[condition_mask]
    y.reset_index(drop = True, inplace = True)
    
    label1 = np.repeat('rest', 366)
    for i in designS:
        label1[i:i+21] = shape 
    for j in designF:
        label1[j:j+21] = face
    labels1 = pd.DataFrame(label1, columns = ['labels'])
    y1 = labels1['labels']
    #Restrict to shapes and face
    #condition_mask1 = y1.isin(['shap', 'face'])
    #condition_mask1 = y1.isin(['rest', 'face'])
    condition_mask1 = y1.isin([con1, con2])
    y1 = y1[condition_mask1]
    y1.reset_index(drop = True, inplace = True)
    return condition_mask, y, condition_mask1, y1
    
condition_mask, y0, condition_mask1, y1 = label_condition('shap', 'face')

def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({'A': A.flatten()}, index=index).reset_index()
    return df

def reshape_fmri(fmri):
    n_samples, n_x, n_y = fmri.shape
    return fmri.reshape((n_samples,n_x*n_y))

def tfMRI_searchlight_raw(n_jobs, n_splits, tfmri_filename, mask_int, C = 0.1):
    cv = KFold(n_splits = n_splits)
    #from sklearn.ensemble import RandomForestClassifier
    model =  LinearSVC(dual = False, max_iter = 10000, C = C)
    tfmri_masked = []
    for i in range(0, tfmri_filename.shape[0]):
        start_time = time.time()
        mask_fmri = mask_int
        searchlight = nilearn.decoding.SearchLight(mask_fmri, n_jobs=n_jobs, verbose=1, cv=cv, estimator = model)
        if (nib.load(tfmri_filename[i]).shape[3] == 366):
            fmri_img = index_img(tfmri_filename[i], condition_mask1)
            searchlight.fit(fmri_img, y1)
        else:
            fmri_img = index_img(tfmri_filename[i], condition_mask)
            searchlight.fit(fmri_img, y)
        score = searchlight.scores_
        df = using_multiindex(score, list('XYZ'))
        dff = df.loc[dff['A'] > 0.5 | dff['A'] == 'nan']
        dff.reset_index(drop = True, inplace = True)
        score_binary = np.zeros_like(score)
        for j in range(dff_sorted.shape[0]):
            score_binary[dff_sorted['X'][j]][dff_sorted['Y'][j]][dff_sorted['Z'][j]] = 1
        score_mask =  new_img_like(mask_fmri, score_binary)
        masker = NiftiMasker(mask_strategy='template', mask_img = score_mask, memory="nilearn_cache", memory_level=1)
        tfmri_masked.append(masker.fit_transform(fmri_img))
        print(i)
        print(tfmri_masked[i].shape)
        print(time.time() - start_time)
    return score_mask, tfmri_masked

m = tfMRI_searchlight_raw(n_jobs = 10, n_splits = 4, tfmri_filename = x_inner_train[0:25], mask_int = mask_int, C = 0.1)

###############################################################################################################################################
filename = pd.read_csv('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/MD_HC_tfMRI.csv', dtype = str)
#y = np.repeat([1, 0], 234).ravel()
outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 6)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 6)
j = 4
train, test = list(outer_cv.split(filename, y))[j]
x_train, x_test = np.array(filename['file'].loc[train]), np.array(filename['file'].loc[test])
y_train, y_test = y[train], y[test]
h = 2
inner_train, inner_test = list(inner_cv.split(x_train, y_train))[h]
x_inner_train, x_inner_test = x_train[inner_train], x_train[inner_test]
y_inner_train, y_inner_test = y_train[inner_train], y_train[inner_test]
##############################################################################################################################################

mask_fmri = nifti_masker.fit(filename['file'].loc[9]).mask_img_
template = load_mni152_template()
def tfMRI_searchlight(n_jobs, n_splits, tfmri_filename, mask_int, C, template, y, y1, condition_mask, condition_mask1):
    cv = KFold(n_splits = n_splits)
    #from sklearn.ensemble import RandomForestClassifier
    model =  LinearSVC(dual = False, max_iter = 10000, C = C)
    tfmri_masked = []
    score_mask = []
    for i in range(0, tfmri_filename.shape[0]):
        start_time = time.time()
        mask_fmri = mask_int
        searchlight = nilearn.decoding.SearchLight(mask_fmri, n_jobs=n_jobs, verbose=1, cv=cv, estimator = model)
        re = resample_to_img(tfmri_filename[i], template)
        if (re.shape[3] == 366):
            fmri_img = index_img(re, condition_mask1)
            searchlight.fit(fmri_img, y1)
        else:
            fmri_img = index_img(re, condition_mask)
            searchlight.fit(fmri_img, y0)
        score = searchlight.scores_
        df = using_multiindex(score, list('XYZ'))
        dff = df.loc[df['A'] > 0.5]
        dff.reset_index(drop = True, inplace = True)
        score_binary = np.zeros_like(score)
        for k in range(dff.shape[0]):
            score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
        score_mask.append(new_img_like(mask_fmri, score_binary))
        print(i)
        print(time.time() - start_time)
    return score_mask
    
start_time1 = time.time()
s = tfMRI_searchlight(n_jobs = 10, n_splits = 4, tfmri_filename = x_inner_train, mask_int = mask_int, C = 0.1, template = template, y = y0, y1 = y1, condition_mask = condition_mask, condition_mask1 = condition_mask1)
print(time.time() - start_time1)

mask_inter_85 = intersect_masks(s, threshold = 0.85)

#############################################################store
mask_int = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMRI468_mask_inter.nii.gz")
template = nib.load(filename['file'].loc[9])
condition_mask, y0, condition_mask1, y1 = label_condition('face', 'rest')
condition_mask1[227:261] = False

cv = KFold(n_splits = 4)
#from sklearn.ensemble import RandomForestClassifier
model =  LinearSVC(dual = False, max_iter = 10000, C = 0.1)
model1 =  LinearSVC(dual = False, max_iter = 10000, C = 0.1)
tfmri_masked = []
score_mask = []
nifti_masker = NiftiMasker(mask_strategy='template', memory="nilearn_cache",  memory_level=1)
mni = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (filename['file'].loc[i][2:9]))
mask_fmri = nifti_masker.fit(mni).mask_img_
searchlight = nilearn.decoding.SearchLight(mask_img = mask_fmri, n_jobs = 10, verbose=1, cv=cv, estimator = model)

for i in range(393, filename.shape[0]):
    start_time = time.time()
    mni = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (filename['file'].loc[i][2:9]))
    if (mni.shape[3] == 366):
        fmri_img = index_img(mni, condition_mask1)
        searchlight.fit(fmri_img, y1)
    else:
        fmri_img = index_img(mni, condition_mask)
        searchlight.fit(fmri_img, y0)
    score = searchlight.scores_
    np.save("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_MNI_05score/%s_score.npy" % (filename['file'].loc[i][2:9]), score)
    print(i)
    print(score.shape)
    print(time.time() - start_time)

    df = using_multiindex(score, list('XYZ'))
    dff = df.loc[df['A'] > 0.5]
    dff.reset_index(drop = True, inplace = True)
    score_binary = np.zeros_like(score)
    for k in range(dff.shape[0]):
        score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
    print(i)
    print(np.unique(score_binary, return_counts = True))
    print(score_binary.shape)
    nib.save(new_img_like(mask_fmri, score_binary), "/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_FR_05score/%s_score.nii.gz" % (filename['file'].loc[i][2:9]))
    print(time.time() - start_time)

score_map = []
for i in range(0, filename.shape[0]):
    score = np.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_MNI_05score/%s_score.npy" % (filename['file'].loc[i][2:9]))
    df = using_multiindex(score, list('XYZ'))
    dff = df.loc[df['A'] > 0.5]
    dff.reset_index(drop = True, inplace = True)
    score_binary = np.zeros_like(score)
    for k in range(dff.shape[0]):
        score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
    score_map.append(new_img_like(mask_fmri, score_binary))
    print("Score mask reading process{0}%".format(round((i+1)*100/x_inner_train.shape[0])), end="\r")
    #print(np.unique(score_binary, return_counts = True))
    #print(score_binary.shape)
    #print(len(score_map))

score_map = np.zeros(template.shape)
for i in range(0, filename.shape[0]):
    score = np.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_MNI_05score/%s_score.npy" % (filename['file'].loc[i][2:9]))
    df = using_multiindex(score, list('XYZ'))
    dff = df.loc[df['A'] > 0.5]
    dff.reset_index(drop = True, inplace = True)
    score_binary = np.zeros_like(score)
    for k in range(dff.shape[0]):
        score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = dff['A'][k]
    score_map = np.add(score_map, score_binary)
    print("Process{0}%".format(round((i+1)*100/filename.shape[0])), end="\r")
    
    score_map.append(new_img_like(template, score_binary))
    print("Score mask reading process{0}%".format(round((i+1)*100/x_inner_train.shape[0])), end="\r")
    #print(np.unique(score_binary, return_counts = True))
    #print(score_binary.shape)
    #print(len(score_map))

searchlight_img = new_img_like(template, score_map)
plotting.plot_img(searchlight_img, bg_img=template, title="Searchlight", display_mode="z", cut_coords=[-9], cmap='hot', threshold = 7, black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/searchlightMNI9_map.png')    
colorbar =  True
searchlight_img = new_img_like(template, StandardScaler().fit_transform(score_map))
plotting.plot_img(searchlight_img, bg_img=template, title="Searchlight", display_mode="z", cut_coords=[-9], cmap='hot', threshold = 1.5, black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/searchlightMNI9_mean.png')    

plotting.plot_anat(t1, title="T1", cut_coords=[-6, -37, -1], black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/5717917T1.png')    
plotting.plot_anat(t1_mni, title="T1 MNI", cut_coords=[-6, -37, -1], black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/5717917T1_MNI.png')    
plotting.plot_roi(mask_T1_mni, title="T1 MNI Mask", black_bg=True, bg_img=t1_mni, cut_coords=[-6, -37, -1], output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/5717917T1_MNI_mask.png')    

plotting.plot_anat(t1_1, title="T1", black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/1002763T1.png')    
plotting.plot_anat(t1_mni_1, title="T1 MNI", cut_coords=[-6, -37, -1], black_bg=True, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/1002763T1_MNI.png')    
plotting.plot_roi(mask_T1_mni_1, title="T1 MNI Mask", black_bg=True, bg_img=t1_mni_1, cut_coords=[-6, -37, -1], output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/1002763T1_MNI_mask.png')    

i = 5
start_time = time.time()
re = resample_to_img(x_test[i], template)
if (re.shape[3] == 366):
    fmri_img = index_img(re, condition_mask1)
    searchlight.fit(fmri_img, y1)
else:
    fmri_img = index_img(re, condition_mask)
    searchlight.fit(fmri_img, y0)

score = searchlight.scores_
df = using_multiindex(score, list('XYZ'))
dff = df.loc[df['A'] > 0.5]
dff.reset_index(drop = True, inplace = True)
score_binary = np.zeros_like(score)
for k in range(dff.shape[0]):
    score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
nib.save(new_img_like(mask_fmri, score_binary), "/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_05score/%s_score.nii.gz" % (x_test[i][2:9]))
print(sum(sum(sum(score_binary))))
print(i)
print(time.time() - start_time)

for i in range(6, x_test.shape[0]):
    start_time = time.time()
    re = resample_to_img(x_test[i], template)
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
        searchlight.fit(fmri_img, y1)
    else:
        fmri_img = index_img(re, condition_mask)
        searchlight.fit(fmri_img, y0)
    score = searchlight.scores_
    df = using_multiindex(score, list('XYZ'))
    dff = df.loc[df['A'] > 0.5]
    dff.reset_index(drop = True, inplace = True)
    score_binary = np.zeros_like(score)
    for k in range(dff.shape[0]):
        score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
    nib.save(new_img_like(mask_fmri, score_binary), "/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_05score/%s_score.nii.gz" % (x_test[i][2:9]))
    print(np.unique(score_binary, return_counts = True))
    print(i)
    print(time.time() - start_time)

for i in range(0, x_inner_train.shape[0]):
    nib.save(s[i], "/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_05score/%s_score.nii.gz" % (x_inner_train[i][2:9]))
    print(i)

score_map = []
for i in range(0, x_inner_train.shape[0]):
    score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_05score/%s_score.nii.gz" % (x_inner_train[i][2:9])))
    print(i)
    
score_map = []
for i in range(0, x_train.shape[0]):
    score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_05score/%s_score.nii.gz" % (x_train[i][2:9])))
    print(i)    

score_map = []
for i in range(0, x_inner_train.shape[0]):
    score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_FR_05score/%s_score.nii.gz" % (x_inner_train[i][2:9])))
    print(i)
    
score_map = []
for i in range(0, x_train.shape[0]):
    score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_FR_05score/%s_score.nii.gz" % (x_train[i][2:9])))
    #print(i)    

percs = [650, 675, 700]

def plot_mask(perc):
    score_int = intersect_masks(score_map, threshold = perc*0.001)
    print(np.unique(score_int.get_data(), return_counts = True))
    plotting.plot_roi(score_int, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/tfMRI_MNI%s_mask4' % (perc))

for perc in percs:
    plot_mask(perc)
    print(perc)

perc = 40
score_int = intersect_masks(score_map, threshold = perc*0.001)
np.unique(score_int.get_data(), return_counts = True)
plotting.plot_roi(score_int, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/tfMRI_SF05_MNI03_40.png')
new_masker = NiftiMasker(mask_img = score_int,  memory="nilearn_cache", memory_level=1)

#plotting.plot_roi(score_int, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/score_map_65%_00.png')

tf_train_masked = []

for i in range(0, x_inner_train.shape[0]):
    start_time = time.time()
    re = resample_to_img(x_inner_train[i], template)
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_train_masked.append(new_masker.fit_transform(fmri_img))
    print(i)
    print(tf_train_masked[i].shape)
    print(time.time() - start_time)

def tf(i):  
    start_time = time.time()
    re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (x_inner_train[i][2:9]))
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_new_masked = new_masker.fit_transform(fmri_img)
    tf_train_masked.append(tf_new_masked)
    print(i)
    print(tf_new_masked.shape)
    print(time.time() - start_time)

def tfoutter(i):  
    start_time = time.time()
    re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (x_train[i][2:9]))
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_new_masked = new_masker.fit_transform(fmri_img)
    tf_train_masked.append(tf_new_masked)
    print(i)
    print(tf_new_masked.shape)
    print(time.time() - start_time)

np.unique(y_inner_train, return_counts = True)
tf_train_masked = []
Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
        delayed(tf)(
            i,
        )
        #for i in range(0, 150)
        for i in range(150, x_inner_train.shape[0])
    )

np.unique(y_train, return_counts = True)
tf_train_masked = []
Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
        delayed(tfoutter)(
            i,
        )
        for i in range(0, 188)
        #for i in range(187, x_train.shape[0])
    )

model =  LinearSVC(dual = False, max_iter = 10000, C = 0.1)
model1 =  LinearSVC(dual = False, max_iter = 10000, C = 1)
model = SVC(kernel = 'linear', probability = True, C = 0.1)
X = reshape_fmri(np.array(tf_train_masked))

clf = make_pipeline(StandardScaler(), model)
start_time = time.time()
clf.fit(X, y_inner_train)
print(time.time() - start_time)

clf1 = make_pipeline(StandardScaler(), model1)
start_time = time.time()
clf1.fit(X, y_inner_train)
print(time.time() - start_time)

clf = make_pipeline(StandardScaler(), model)
start_time = time.time()
clf.fit(X, y_train)
print(time.time() - start_time)

clf1 = make_pipeline(StandardScaler(), model1)
start_time = time.time()
clf1.fit(X, y_inner_train)
print(time.time() - start_time)

scaler = StandardScaler()
X1 = scaler.fit_transform(X)
print(scaler.mean_)

X2 = np.around(X1, decimals=6)

start_time = time.time()
model.fit(X1, y_inner_train)
print(time.time() - start_time)

start_time = time.time()
model1.fit(X1, y_inner_train)
print(time.time() - start_time)

start_time = time.time()
model.fit(X, y_train)
print(time.time() - start_time)

start_time = time.time()
model1.fit(X1, y_train)
print(time.time() - start_time)

def tftest(i):  
    start_time = time.time()
    re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (x_inner_test[i][2:9]))
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_new_masked = new_masker.fit_transform(fmri_img)
    tf_test_masked.append(tf_new_masked)
    print(i)
    print(tf_new_masked.shape)
    print(time.time() - start_time)
    
def tfouttertest(i):  
    start_time = time.time()
    re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (x_test[i][2:9]))
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_new_masked = new_masker.fit_transform(fmri_img)
    tf_test_masked.append(tf_new_masked)
    print(i)
    print(tf_new_masked.shape)
    print(time.time() - start_time)

np.unique(y_inner_test, return_counts = True)

tf_test_masked = []
Parallel(n_jobs = 20, verbose = 0, require='sharedmem')(
        delayed(tftest)(
            i,
        )
        #for i in range(0, 38)
        for i in range(38, x_inner_test.shape[0])
    )
    
np.unique(y_test, return_counts = True)
tf_test_masked = []
Parallel(n_jobs = 15, verbose = 0, require='sharedmem')(
        delayed(tfouttertest)(
            i,
        )
        #for i in range(0, 47)
        for i in range(47, x_test.shape[0])
    )
    
for i in range(0, x_inner_test.shape[0]):
    start_time = time.time()
    re = resample_to_img(x_inner_test[i], template)
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_test_masked.append(new_masker.fit_transform(fmri_img))
    print(i)
    print(tf_test_masked[i].shape)
    print(time.time() - start_time)

Y = reshape_fmri(np.array(tf_test_masked))

start_time = time.time()
y_inner_pred = clf.predict(Y)
print(time.time() - start_time)

start_time = time.time()
y_pred = clf1.predict(Y)
print(time.time() - start_time)

Y1 = scaler.transform(Y)

start_time = time.time()
y_inner_pred = model.predict(Y1)
print(time.time() - start_time)

start_time = time.time()
y_inner1_pred = model1.predict(Y1)
print(time.time() - start_time)

start_time = time.time()
y_pred = model.predict(Y1)
print(time.time() - start_time)

start_time = time.time()
y_pred1= model1.predict(Y1)
print(time.time() - start_time)

metrics.accuracy_score(y_inner_test, y_inner_pred)
metrics.f1_score(y_inner_test, y_inner_pred)

metrics.accuracy_score(y_inner_test, y_inner1_pred)
metrics.f1_score(y_inner_test, y_inner1_pred)

metrics.accuracy_score(y_test, y_pred)
metrics.f1_score(y_test, y_pred)

metrics.accuracy_score(y_test, y_pred1)
metrics.f1_score(y_test, y_pred)

start_time = time.time()
y_pred = clf1.predict(Y)
print(time.time() - start_time)

metrics.accuracy_score(y_test, y_pred)
metrics.f1_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred, average='binary')
metrics.precision_score(y_test, y_pred, average='binary')
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
s = tn / (tn+fp)
print(s)
y_score = clf1.predict_proba(Y)[:,1]
pre, re, thresholds = precision_recall_curve(y_test, y_score)
auc(re, pre)
fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False)
auc(fpr, tpr)

-------------------------------------------------------------------------------------------------------------------------------------
def reshape_fmri(fmri):
    n_samples, n_x, n_y = fmri.shape
    return fmri.reshape((n_samples,n_x*n_y))

new_masker = NiftiMasker(mask_img = score_int,  memory="nilearn_cache", memory_level=1)

def tfMRI_mask_train(i, data, label,condition_mask1, condition_mask, new_masker):  
re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (data[i][2:9]))
if (re.shape[3] == 366):
    fmri_img = index_img(re, condition_mask1)
else:
    fmri_img = index_img(re, condition_mask)    

tf_new_masked = new_masker.fit_transform(fmri_img)
tf_train_masked.append(tf_new_masked)
tf_train_label.append(label[i])
    print("Train mask process{0}%".format(round((i+1)*100/len(data))), end="\r")

def tfMRI_mask_test(i, data, label, condition_mask1, condition_mask, new_masker):  
    start_time = time.time()
    re = nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/tfMNI_resample/%s_MNI.nii.gz" % (data[i][2:9]))
    if (re.shape[3] == 366):
        fmri_img = index_img(re, condition_mask1)
    else:
        fmri_img = index_img(re, condition_mask)
    tf_new_masked = new_masker.fit_transform(fmri_img)
    tf_test_masked.append(tf_new_masked)
    tf_test_label.append(label[i])
    print("Test mask process{0}%".format(round((i+1)*100/len(data))), end="\r")
-----------------------
score_map = []
for i in range(0, filename.shape[0]):
    score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_FR_05score/%s_score.nii.gz" % (filename['file'].loc[i][2:9])))
    print("Score mask reading process{0}%".format(round((i+1)*100/filename.shape[0])), end="\r")
 
score_int = intersect_masks(score_map, threshold = 0.60)
np.unique(score_int.get_data(), return_counts = True)
-----------------------
filename = pd.read_csv('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/MD_HC_tfMRI.csv', dtype = str)
#y = np.repeat([1, 0], 234).ravel()
outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 12)
j = 0
train, test = list(outer_cv.split(filename, y))[j]
x_train, x_test = np.array(filename['file'].loc[train]), np.array(filename['file'].loc[test])
y_train, y_test = y[train], y[test]
a = []
f11 = []
for h in range(3, 5):
    inner_train, inner_test = list(inner_cv.split(x_train, y_train))[h]
    x_inner_train, x_inner_test = x_train[inner_train], x_train[inner_test]
    y_inner_train, y_inner_test = y_train[inner_train], y_train[inner_test]
    score_map = []
    for i in range(0, x_inner_train.shape[0]):
        score_map.append(nib.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_FR_05score/%s_score.nii.gz" % (x_inner_train[i][2:9])))
        print("Score mask reading process{0}%".format(round((i+1)*100/x_inner_train.shape[0])), end="\r")
    score_int = intersect_masks(score_map, threshold = 0.675)
    new_masker = NiftiMasker(mask_img = score_int,  memory="nilearn_cache", memory_level=1)
    condition_mask = np.repeat(True, 332).ravel()
    condition_mask1 = np.repeat(True, 366).ravel()
    condition_mask1[332:] = False
    tf_train_masked = []
    tf_train_label = []
    Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
        delayed(tfMRI_mask_train)(
            i, x_inner_train, y_inner_train, condition_mask1, condition_mask,
        )
        for i in range(0, x_inner_train.shape[0])
    )
    tf_test_masked = []
    tf_test_label = []
    Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
        delayed(tfMRI_mask_test)(
            i, x_inner_test, y_inner_test, condition_mask1, condition_mask,
        )
        for i in range(0, x_inner_test.shape[0])
    )
    #model =  RandomForestClassifier()
    model =  LinearSVC(dual = False, max_iter = 10000, C = 0.1)
    X = reshape_fmri(np.array(tf_train_masked))
    clf = make_pipeline(StandardScaler(), model)
    start_time = time.time()
    clf.fit(X, tf_train_label)
    print("Training time", time.time() - start_time)
    Y = reshape_fmri(np.array(tf_test_masked))
    start_time = time.time()
    y_inner_pred = clf.predict(Y)
    print("Testing time", time.time() - start_time)
    a1 = metrics.accuracy_score(tf_test_label, y_inner_pred)
    f11 = metrics.f1_score(tf_test_label, y_inner_pred)
    a.append(a1)
    f.append(f11)

inner_train, inner_test = list(inner_cv.split(x_train, y_train))[h]
x_inner_train, x_inner_test = x_train[inner_train], x_train[inner_test]
y_inner_train, y_inner_test = y_train[inner_train], y_train[inner_test]
score_map = []
for i in range(0, x_inner_train.shape[0]):
    score = np.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_MNI_05score/%s_score.npy" % (x_inner_train[i][2:9]))
    df = using_multiindex(score, list('XYZ'))
    dff = df.loc[df['A'] > 0.5]
    dff.reset_index(drop = True, inplace = True)
    score_binary = np.zeros_like(score)
    for k in range(dff.shape[0]):
        score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
    score_map.append(new_img_like(mask_fmri, score_binary))
    print("Score mask reading process{0}%".format(round((i+1)*100/x_inner_train.shape[0])), end="\r")
    #print(np.unique(score_binary, return_counts = True))
    #print(score_binary.shape)
    #print(len(score_map))

perc = 35

score_int = intersect_masks(score_map, threshold = perc*0.001)
np.unique(score_int.get_data(), return_counts = True)
plotting.plot_roi(score_int, output_file = '/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/plot_data/tfMRI_SF05_MNI03_%s.png' % perc)
new_masker = NiftiMasker(mask_img = score_int,  memory="nilearn_cache", memory_level=1)
condition_mask = np.repeat(True, 332).ravel()
condition_mask1 = np.repeat(True, 366).ravel()
condition_mask1[332:] = False
tf_train_masked = []
tf_train_label = []
Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
    delayed(tfMRI_mask_train)(
        i, x_inner_train, y_inner_train, condition_mask1, condition_mask,
    )
    for i in range(0, x_inner_train.shape[0])
)
tf_test_masked = []
tf_test_label = []
Parallel(n_jobs = 30, verbose = 0, require='sharedmem')(
    delayed(tfMRI_mask_test)(
        i, x_inner_test, y_inner_test, condition_mask1, condition_mask,
    )
    for i in range(0, x_inner_test.shape[0])
)
model =  RandomForestClassifier()
model =  LinearSVC(dual = False, max_iter = 10000, C = 0.1)
model =  LinearSVC(C = 1)
#model = SVC(kernel = "linear", C = C)
X = reshape_fmri(np.array(tf_train_masked))
p = [10, 100, 1000]
p = [1, 3, 5, 7]
for i in range(len(p)):
    model =  SVC(kernel = "poly", C = 1, degree = p[i])
    print(p[i])
    clf = make_pipeline(StandardScaler(), model)
    start_time = time.time()
    clf.fit(X, tf_train_label)
    print("Training time", time.time() - start_time)
    #Y = reshape_fmri(np.array(tf_test_masked))
    #start_time = time.time()
    y_inner_pred = clf.predict(Y)
    #print("Testing time", time.time() - start_time)
    print(metrics.accuracy_score(tf_test_label, y_inner_pred))
    print(metrics.f1_score(tf_test_label, y_inner_pred))

warnings.filterwarnings("ignore")
polydegree = [1, 3, 5, 7]
a = []
f = []
outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 6)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 6)
y = np.array(filename['label']).ravel()
for j in range(5):
    train, test = list(outer_cv.split(filename, y))[j]
    x_train, x_test = np.array(filename['file'].loc[train]), np.array(filename['file'].loc[test])
    y_train, y_test = y[train], y[test]
    template = load_mni152_template()
    score_map = []
    for i in range(0, x_train.shape[0]):
        score = np.load("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/Searchlight_SF_MNI_05score/%s_score.npy" % (x_train[i][2:9]))
        df = using_multiindex(score, list('XYZ'))
        dff = df.loc[df['A'] > 0.5]
        dff.reset_index(drop = True, inplace = True)
        score_binary = np.zeros_like(score)
        for k in range(dff.shape[0]):
            score_binary[dff['X'][k]][dff['Y'][k]][dff['Z'][k]] = 1
        score_map.append(new_img_like(template, score_binary))
        print("Score mask reading process{0}%".format(round((i+1)*100/x_train.shape[0])), end="\r")
    score_int = intersect_masks(score_map, threshold = 0.03)
    np.unique(score_int.get_data(), return_counts = True)
    new_masker = NiftiMasker(mask_img = score_int,  memory="nilearn_cache", memory_level=1)
    condition_mask = np.repeat(True, 332).ravel()
    condition_mask1 = np.repeat(True, 366).ravel()
    condition_mask1[332:] = False
    tf_train_masked = []
    tf_train_label = []
    Parallel(n_jobs = 20, verbose = 0, require='sharedmem')(
        delayed(tfMRI_mask_train)(
            i, x_train, y_train, condition_mask1, condition_mask, new_masker
        )
        for i in range(0, x_train.shape[0])
    )
    np.save('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/searchlight_tfmri_SF_mni_masked/tf_train%s_masked.npy' % (j),  np.asarray(tf_train_masked))
    np.save('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/searchlight_tfmri_SF_mni_masked/tf_train%s_label.npy' % (j),  np.asarray(tf_train_label))
    tf_test_masked = []
    tf_test_label = []
    Parallel(n_jobs = 20, verbose = 0, require='sharedmem')(
        delayed(tfMRI_mask_test)(
            i, x_test, y_test, condition_mask1, condition_mask, new_masker
        )
        for i in range(0, x_test.shape[0])
    )
    np.save('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/searchlight_tfmri_SF_mni_masked/tf_test%s_masked.npy' % (j),  np.asarray(tf_test_masked))
    np.save('/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/searchlight_tfmri_SF_mni_masked/tf_test%s_label.npy' % (j),  np.asarray(tf_test_label))
    
a = []
f = []
for l in range(len(polydegree)):
    model = SVC(kernel = "poly", degree = polydegree[l])
    X = reshape_fmri(np.array(tf_train_masked))
    clf = make_pipeline(StandardScaler(), model)
    start_time = time.time()
    clf.fit(X, tf_train_label)
    print("Training time", time.time() - start_time)
    Y = reshape_fmri(np.array(tf_test_masked))
    start_time = time.time()
    y_pred = clf.predict(Y)
    print("Testing time", time.time() - start_time)
    a.append(metrics.accuracy_score(tf_test_label, y_pred))
    f.append(metrics.f1_score(tf_test_label, y_pred))

print(np.mean(a))
print(np.mean(f))
        

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'degree':[1, 3, 5, 7], 'C': [0.1, 1, 10, 100]}]

scores = ['precision', 'recall']

start_time = time.time()
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    model = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X, tf_train_label)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = tf_test_label, clf.predict(Y)
    print(classification_report(y_true, y_pred))
    print()
print("Testing time", time.time() - start_time)
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
