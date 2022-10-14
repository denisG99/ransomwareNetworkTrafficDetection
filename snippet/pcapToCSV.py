'''
Script che permette la conversione del file pcap in un file CSV atteaverso il modulo di wireshark "tshark".
L'header CSV ha il seguente formato:
(frame.number, frame.time, ip.src, tcp.srcport, udp.srcport, id.dst, tcp.dstport, udp.dstport, ip.proto, frame.len, tcp.time_delta, udp.time_delta)
'''

import os

RAWDATA_PATH = "../dataset_download/raw/"  #percorso in cui posso trovare i pcap da convertire

def pcap_to_csv(path, dest_path, lst_filter):
    #TODO: vedere se c'è un modo per estrarre tabella delle comunicazioni (statistica wireshark)
    cmd = f"tshark -r {path} -T fields {build_filter(lst_filter)} -E header=y -E separator=, -E quote=d > {dest_path}"

    os.system(cmd)

def build_filter(lst):
    res = ""
    for elem in lst:
        res += f" -e {elem}"

    return res

for d in os.listdir(f"{RAWDATA_PATH}/pcap/"):
    for pcap in os.listdir(f"{RAWDATA_PATH}/pcap/{d}"): #contiene lista pcap da convertire in formato CSV
        # stringa contenente comando che permette la conversione da formato pcap a csv da linea di comando usando l'utility tshark (componenete di wireshark)
        pcapPath = f"{RAWDATA_PATH}/pcap/{d}/{pcap}"
        dest = f"{RAWDATA_PATH}csv/{d}/{pcap.replace('.pcap', '')}.csv"
        filters = ['frame.number', 'frame.time', 'ip.src', 'tcp.srcport', 'udp.srcport', 'ip.dst', 'tcp.dstport', 'udp.dstport',
                   'ip.proto', 'frame.len', 'tcp.time_delta', 'udp.time_delta', 'udp.checksum', 'tcp.checksum', ]

        pcap_to_csv(pcapPath, dest, filters)
