import numpy as np

def save_q_table(Q_table, filepath):
    np.save(filepath, Q_table)

def load_q_table(filepath):
    return np.load(filepath)
