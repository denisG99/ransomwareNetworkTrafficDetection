'''
Script che permette la conversione del file pcap in un file CSV atteaverso il modulo di wireshark "tshark".
L'header CSV ha il seguente formato:
(frame.number, frame.time, ip.src, tcp.srcport, udp.srcport, id.dst, tcp.dstport, udp.dstport, ip.proto, frame.len, tcp.time_delta, udp.time_delta)
'''

import os

RAWDATA_PATH = "../dataset/raw/" #percorso in cui posso trovare i pcap da convertire

for d in os.listdir(f"{RAWDATA_PATH}/pcap/"):
    for pcap in os.listdir(f"{RAWDATA_PATH}/pcap/{d}"): #contiene lista pcap da convertire in formato CSV
        # stringa contenente comando che permette la conversione da formato pcap a csv da linea di comando usando l'utility tshark (componenete di wireshark)
        pcapPath = f"{RAWDATA_PATH}/pcap/{d}/{pcap}"
        dest = f"{RAWDATA_PATH}csv/{d}/{pcap.replace('.pcap', '')}.csv"
    
        cmd = f"tshark -r {pcapPath} -T fields -e frame.number -e frame.time -e ip.src -e tcp.srcport -e udp.srcport " \
              f"-e ip.dst -e tcp.dstport -e udp.dstport -e ip.proto -e frame.len -e tcp.time_delta -e udp.time_delta " \
              f"-E header=y -E separator=, -E quote=d > {dest}"
    
        os.system(cmd)
