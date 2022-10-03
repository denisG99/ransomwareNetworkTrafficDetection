import os
from Logger import Logger
#modulo per decompressione file zip(.zip) e gzip(.gz)
from zipfile import ZipFile
import shutil
import gzip as gz

# path ARCHIVIO
ARCHIVE_PATH_GOODWARE = "/Volumes/ARCHIVIO/goodware"
ARCHIVE_PATH_MALWARE = "/Volumes/ARCHIVIO/malware"
ARCHIVE_DEST_PATH = "/Volumes/ARCHIVIO/csv"

logger = Logger('FEATURES EXTRACTION FAIL', 'FEATURES EXTRACTION SUCCESS', 'extraction.log')

###################################################### FUNCTIONS ######################################################

def feature_extract(path, traffic_type):
    for file in os.listdir(path):
        if not os.path.isfile(f"{path}/{file}"):
            feature_extract(f"{path}/{file}", traffic_type)

        if file.endswith(".pcapng"):
            cmd = f"tshark -F pcap -r {path}/{file} -w {path}/{file.replace('.pcapng', '.pcap')}"

            os.system(cmd)
            os.remove(f"{path}/{file}")
        elif file.endswith(".pcap"):
            #cmd = f"cicflowmeter -f {path}/{file} -c {ARCHIVE_DEST_PATH}/{traffic_type}/{file.replace('.pcap', '.csv')}"
            cmd = f"cicflowmeter -f ../prova/{file} -c ../prova/{traffic_type}/{file.replace('.pcap', '.csv')}"
            status_code = os.system(cmd)

            logger.log_writer(f"{path}/{file}", status_code)

def unzip_pcap(path, type, pswd = None):
    splitted_path = path.split('/')

    dest_dir = "/".join(splitted_path[ : len(splitted_path) - 1])

    if type == 'zip':
        with ZipFile(path, 'r') as obj_zip:
            for file in obj_zip.namelist():
                if file.endswith('.pcap'): #necessaria perchè ci sono certi archivi che contengono alti file inerenti al ransonware oltre al pcap, che è ciò che mi interessa
                    obj_zip.extract(file, f'{dest_dir}/', pwd = bytes(pswd, 'utf-8'))
    elif type == 'gz':
        dest_name = splitted_path[-1]

        with gz.open(path, 'rb') as f_in:
            with open(f'{dest_dir}/{dest_name[: len(dest_name) - 3]}', 'wb') as f_out:
               shutil.copyfileobj(f_in, f_out)

def main():
    #feature_extract(ARCHIVE_PATH_GOODWARE, 'goodware')
    #feature_extract(ARCHIVE_PATH_MALWARE, 'malware')
    # TODO: rivedere path
    for archive in os.listdir('../prova'):
        if not archive == ".DS_Store":
            print(f"Decompressione {archive}...")
            if archive.endswith('zip'):
                unzip_pcap(f'../prova/{archive}', pswd='infected', type='zip')
            elif archive.endswith('gz'):
                unzip_pcap(f'../prova/{archive}', type='gz')

    feature_extract("../prova", 'malware')


if __name__ == "__main__":
    main()