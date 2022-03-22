'''
fa sempre par della fase di pre-processing: cerco le informazioni inernti alla connessione

si parte dal presupposto che ogni comunicazzione avvenga in maniera bidirezionale, tendenzialmete è così
'''

import pandas as pd

CSV_PATH = "../dataset/raw/csv/prova.csv"

def get_connections(data):
    connections = list()

    for record in data.itertuples():
        conn = (record[4], record[5], record[6], record[7], record[8]) #tupla contenente la connessione

        if conn not in connections:
            connections.append(conn)

    return connections

df = pd.read_csv(CSV_PATH)
conns = get_connections(df)
prev_conn = None
Bytes, Duration, src2dstPcks, src2dstBytes, dst2srcPcks, dst2srcBytes = list(), list(), list(), list(), list(), list()
i = 0
print(conns)
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
        Duration.append(aux['Duration'].sum())
        src2dstPcks.append(len(aux))
        src2dstBytes.append(aux['Length'].sum())

        prev_conn = conn

    else:
        if src_ip == prev_conn[2] and src_port == prev_conn[3] and dst_ip == prev_conn[0] and dst_port == prev_conn[1] and proto == prev_conn[4]:
            Bytes[i] += aux['Length'].sum()
            Duration[i] += aux['Duration'].sum()
            dst2srcPcks.append(len(aux))
            dst2srcBytes.append(aux['Length'].sum())

        i += 1
        prev_conn = None

#print(Bytes, Duration, src2dstPcks, src2dstBytes, dst2srcPcks, dst2srcBytes)

aggregateData = pd.DataFrame({"Bytes": Bytes,
                              "Duration": Duration,
                              "src2dstPcks": src2dstPcks,
                              "src2dstBytes": src2dstBytes,
                              "dst2srcPcks": dst2srcPcks,
                              "dst2srcBytes": dst2srcBytes})

aggregateData.to_csv('../dataset/raw/csv/aggregate.csv', index=False)