import numpy as np
import pandas as pd

labels = [
    "job_id", "submit_time", "wait_time", "run_time", "allocated_proc",
    "cpu_time", "used_mem", "requested_proc", "requested_time", "requested_mem",
    "status", "user_id", "group_id", "executable_id", "queue_id",
    "partition_id", "preceding_job_id", "think_time"
]


TOTAL_CORES = 128
INTERVAL_SEC = 3600  
SLOTS_PER_DAY = 24

max_time = (df["submit_time"] + df["run_time"]).max()
total_slots = int(np.ceil(max_time / INTERVAL_SEC))
active_cores = np.zeros(total_slots)

#check active cores per interval for predicting workloads
for _, row in df.iterrows():
    start_slot = int(row["submit_time"] // INTERVAL_SEC)
    end_slot = int((row["submit_time"] + row["run_time"]) // INTERVAL_SEC)
    procs = row["allocated_proc"]
    
    if procs > 0:
        active_cores[start_slot : end_slot + 1] += procs

#Normalize workload (x^{d,t}) from now on below this reference page 4 of the paper
workload = np.clip(active_cores / TOTAL_CORES, 0.0, 1.0)

#turning to into 2D Grid e.g Days, Slots_per_day
num_days = len(workload) // SLOTS_PER_DAY
workload_2d = workload[: num_days * SLOTS_PER_DAY].reshape((num_days, SLOTS_PER_DAY))

#2D-LSTM Features implementation (X^t)
def build_2d_features(workload_matrix, n_days=3, m_slots=3):
    X, Y = [], []
    num_days, num_slots = workload_matrix.shape
    
    for d in range(n_days, num_days):
        for t in range(m_slots, num_slots - 1):
            #day-dimensional historical features at time t
            day_features = workload_matrix[d - n_days : d + 1, t]
            
            #time-dimensional historical features on day d
            time_features = workload_matrix[d, t - m_slots : t + 1]
            
            #combine it
            X_t = np.concatenate([day_features, time_features])
            
            #the TARGET is the workload at next interval (t + 1) following the paper
            Y_t = workload_matrix[d, t + 1]
            
            X.append(X_t)
            Y.append(Y_t)
            
    return np.array(X), np.array(Y)

X, Y = build_2d_features(workload_2d, n_days=3, m_slots=3)
print("Feature matrix shape:", X.shape)  
print("Target vector shape:", Y.shape)    