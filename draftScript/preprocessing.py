'''
  * conversione dell'indirizzo IP in numero;
  * aggregare le porta (TCP o UDP) in una singola colonna
  * aggregare i delta (TCP o UDP) in una singola colonna
'''

import pandas as pd
import numpy as np
import os

def set_port(rawData, type): #controllare valori di type (src o dst)
    tcp = np.array(rawData[f'tcp.{type}port'])
    udp = np.array(rawData[f'udp.{type}port'])
    ports = np.zeros(len(rawData))

    for i in range(0, len(tcp)):
        if pd.isna(tcp[i]):
            ports[i] = udp[i]
        elif pd.isna(udp[i]):
            ports[i] = tcp[i]
        elif pd.isna(tcp[i]) and pd.isna(udp[i]):
            ports[i] = 0

    return ports

def set_duration(rawData):
    tcp = np.array(rawData['tcp.time_delta'])
    udp = np.array(rawData['udp.time_delta'])
    deltas = np.zeros(len(rawData))

    for i in range(0, len(rawData)):
        if pd.isna(tcp[i]):
            deltas[i] = udp[i]
        elif pd.isna(udp[i]):
            deltas[i] = tcp[i]
        elif pd.isna(tcp[i]) and pd.isna(udp[i]):
            deltas[i] = 0.0

    return deltas

def IP_to_integer(data):
    ips = np.zeros(len(data))

    for i in range(0, len(data)):
        if pd.isna(data[i]):
            ips[i] = int("0000")
        else:
            ips[i] = int(data[i].replace('.', ''))

    return ips


CSV_PATH = "../dataset/raw/csv/"
csvs = os.listdir(CSV_PATH)

for file in csvs:
    #caso da gestire: caso in cui file CSV sia vuoto
    df = pd.read_csv(f"{CSV_PATH + file}")

    cleaData = pd.DataFrame({"Label" : np.zeros(len(df)), #TODO: da sostituire con algoritmo di etichettatura
                            "Number" : df['frame.number'].to_numpy(),
                            "Time" : df['frame.time'].to_numpy(),
                            "SrcIP" : IP_to_integer(df['ip.src']),
                            "SrcPort" : set_port(df, 'src'),
                            "DstIP" : IP_to_integer(df['ip.dst']),
                            "DstPort" : set_port(df, 'dst'),
                            "Protocol" : df['ip.proto'].to_numpy(),
                            "Length" : df['frame.len'].to_numpy(),
                            "Duration" : set_duration(df)})

    cleaData.to_csv(f'{CSV_PATH}prova.csv', index=False)