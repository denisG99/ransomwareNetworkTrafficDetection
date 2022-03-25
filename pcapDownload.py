import pysftp as sftp
import os

HOST = '142.104.64.196'
PORT = 22
USERNAME = 'gasparollo'
PSW = '' #TODO: da inserire
PVT_KEY_PATH = '/Users/denisgasparollo/.ssh/id_rsa'
PVT_KEY_PSW = '' #inserire nel caso in cui la chiave privata ssh sia protetta da password
DOWNLOAD_LOCAL_PATH = '/Users/denisgasparollo/Desktop/dataset'

def download(conn, remote_path, extension, local_path):
    log_file = open('dwn_log.txt', mode='a')

    for file in conn.listdir:
        if file.isdir(f'{remote_path + file}'):
            log_file.write(f'{file}: \n\t')
            new_local = os.path.join('/Users/denisgasparollo/Desktop/dataset', file)
            os.mkdir(new_local)
            log_file.close()
            return download(conn, f'{remote_path + file}', extension, new_local)
        else:
            if file[-len(extension):] == '.pcap':
                try:
                    conn.get(file, local_path)
                    print(file, ' downloaded successfully ')
                    log_file.write(f'{file}: successfully\n')
                except IOError:
                    print(file, ' downloaded failed ')
                    log_file.write(f'{file}: failed\n')

                return None

    log_file.close()

try:
    conn = sftp.Connection(host=HOST, port= PORT, username=USERNAME, password=PSW, private_key=PVT_KEY_PATH, private_key_pass=PVT_KEY_PSW)
    print('Connessione riuscita')
except :
    print('Connessione non riuscita')

download(conn, conn.getcwd(), '.pcap', DOWNLOAD_LOCAL_PATH)

conn.close()