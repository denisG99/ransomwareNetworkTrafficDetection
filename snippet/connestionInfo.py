"""
fa sempre par della fase di pre-processing: cerco le informazioni inernti alla connessione

si parte dal presupposto che ogni comunicazzione avvenga in maniera bidirezionale, tendenzialmete è così
"""

import pandas as pd
import os
import math

CSV_PATH = "../capture_download/data/"

def get_connections(data):
    connections = list()

    for record in data.itertuples():
        conn = (record[2], record[3], record[4], record[5], record[6]) #tupla contenente la connessione

        if conn not in connections:
            connections.append(conn)

    return connections

for csv in os.listdir(CSV_PATH):
    df = pd.read_csv(f'{CSV_PATH + csv}')
    conns = get_connections(df)
    prev_conn = None
    Bytes, Pcks, Duration, src2dstPcks, src2dstBytes, dst2srcPcks, dst2srcBytes = list(), list(), list(), list(), list(), list(), list()
    i = 0

    for conn in conns:
        src_ip, src_port, dst_ip, dst_port, proto = conn[0], conn[1], conn[2], conn[3], conn[4]

        query = f"SrcIP == {src_ip} " \
                f"and DstIP ==  {dst_ip} " \
                f"and SrcPort == {src_port} " \
                f"and DstPort == {dst_port} " \
                f"and Protocol == {proto}"
        aux = df.query(query)

        if prev_conn is None:
            Bytes.append(aux['Length'].sum())
            Pcks.append(len(aux))
            Duration.append(aux['Duration'].sum())
            src2dstPcks.append(len(aux))
            src2dstBytes.append(aux['Length'].sum())

            prev_conn = conn

        else:
            Bytes[i] += aux['Length'].sum()
            Pcks[i] += len(aux)
            Duration[i] += aux['Duration'].sum()

            if src_ip == prev_conn[2] and src_port == prev_conn[3] and dst_ip == prev_conn[0] and dst_port == prev_conn[1] and proto == prev_conn[4]:
                dst2srcPcks.append(len(aux))
                dst2srcBytes.append(aux['Length'].sum())
            else:
                dst2srcPcks.append(0)
                dst2srcBytes.append(0)

            i += 1
            prev_conn = None

    aggregateData = pd.DataFrame({"Label": list(csv.split(sep='_', maxsplit=2)[0] for i in range(math.ceil(len(conns)/2))), #posso prendere come riferimento un qualsisi vettore
                                  "SrcIP": list(conns[i][0] for i in range(0, len(conns), 2)),
                                  "SrcPort": list(conns[i][1] for i in range(0, len(conns), 2)),
                                  "DstIP": list(conns[i][2] for i in range(0, len(conns), 2)),
                                  "DstPort": list(conns[i][3] for i in range(0, len(conns), 2)),
                                  "Protocol": list(conns[i][4] for i in range(0, len(conns), 2)),
                                  "Bytes": Bytes,
                                  "Packages": Pcks,
                                  "Duration": Duration,
                                  "src2dstPcks": src2dstPcks,
                                  "src2dstBytes": src2dstBytes,
                                  "dst2srcPcks": dst2srcPcks,
                                  "dst2srcBytes": dst2srcBytes})

    aggregateData.to_csv(f'../capture_download/connection/{csv}', index=False)
