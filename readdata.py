import pandas as pd
import numpy as np
import nilearn as nil
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import intersect_masks
from pymatch.Matcher import Matcher



class readata():
    def _init_(self, ND_path, HC_path, modal_path):
        self.ND_path = ND_path
        self.HC_path = HC_path
        self.modal_path = modal_path
    
    def concat(self):
        ND = pd.read_csv(self.ND_path, dtype = str, names = ['col'])
        HC = pd.read_csv(self.HC_path, dtype = str, names = ['col'])
        MD = pd.read_csv("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/phenotype/MD.csv", dtype = str)
        BD = pd.read_csv("/links/groups/borgwardt/Projects/ZhiYe_MasterThesis/phenotype/BD.csv", dtype = str)


        modal = pd.read_csv(self.modal, dtype = str, names = ['file'], header = None)
        modal['col'] = modal['file'].str[2:9]
        
    def PSM()
    
        df = pd.concat([pd.merge(modal, ND, on = 'col'), pd.merge(modal, HC, on = 'col')])
        
        masker = NiftiMasker(mask_strategy='template', memory="nilearn_cache", 
                           memory_level=2)
        
        mri_mask = []
        for i in range(0, filename.shape[0]):
            masker.fit(df['modal'].loc[i])
            mask = masker.mask_img_
            mri_mask,append(mask)
        
        mask_intersec = intersect_masks(mri_mask)
        
        mri_masked = []
        for i in range(0, filename.shape[0]):
            mri_masked = [masker.fit_transform(df['modal'].loc[i], mask_img = mask_intersec), df['col']]
            mri.append(mri_masked)
            
    return mri
