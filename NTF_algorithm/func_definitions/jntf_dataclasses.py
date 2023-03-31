from dataclasses import dataclass
import numpy as np

@dataclass
class JntfDocumentTensors:
    terms: list
    X1:np.ndarray
    time_to_doc1: list
    doc_to_time1: list
    X2:np.ndarray
    time_to_doc2: list
    doc_to_time2: list


# @dataclass
# class JntfResultMatrices:
#     U: list
#     V: list
#     W: list
#     memo:list
