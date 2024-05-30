from typing import *
import pandas as pd
import numpy as np
class CustomDataTransformer:
    def __init__(self, dataset : pd.DataFrame):
        self.dataset = dataset
        self.categorical_columns, self.numerical_columns = self._column_types()
        self.label_values = {}
        self.min_max_values = {}
    
    def _column_types(self) -> List[str]:
        all_columns = list(self.dataset.columns)
        num_col = []
        cat_col = []
        for column in all_columns:
            if type(self.dataset[column][0] )!= str:
                num_col.append(column)
            else:
                cat_col.append(column)
        return cat_col, num_col
    
    def _get_min_and_max(self):
        for col in self.numerical_columns:
            _max = self.dataset[col].max()
            _min = self.dataset[col].min()
            self.min_max_values[f'{col}_min_max'] = [_min, _max]
    
    def _scale(self):
        for col in self.numerical_columns:
            min_max_val = self.min_max_values[f'{col}_min_max']
            self.dataset[col] = self.dataset[col].apply(lambda x: ((x-min_max_val[0])/(min_max_val[1]-min_max_val[0])))
    
    #build standard scaler on your own

    def _label_mapper(self):
        for col in self.categorical_columns:
            uniques_ = list(self.dataset[col].unique())
            j = 0
            self.label_values[f'{col}_label'] = {}
            for i in uniques_:
                self.label_values[f'{col}_label'][f'{i}'] = j
                j += 1 

    def _label_encode(self):
        for col in self.categorical_columns:
            self.dataset[col] = self.dataset[col].apply(lambda x: self.label_values[f'{col}_label'][f'{x}'])


    def fit_transform(self) -> pd.DataFrame:
        self._get_min_and_max()
        self._label_mapper()
        self._scale()
        self._label_encode()
        return self.dataset

    def transform(self, data : dict) -> dict:
        res_dict = {}
        for key, val in data.items():
            if key in self.numerical_columns:
                min_max_val = self.min_max_values[f'{key}_min_max']
                val = ((val-min_max_val[0])/(min_max_val[1]-min_max_val[0]))
                res_dict[key] = val
            else:
                val = self.label_values[f'{key}_label'][f'{val}']
                res_dict[key] = val
        return res_dict