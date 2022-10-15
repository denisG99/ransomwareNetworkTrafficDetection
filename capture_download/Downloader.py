from threading import Thread
import requests as req
import os
import datetime
from bs4 import BeautifulSoup
import queue

class Downloader(Thread):
    def __init__(self, request_queue):
        Thread.__init__(self)
        self.queue = request_queue
        #self.results = []
        self.LOG_FAIL = "\tDownload failed"
        self.LOG_SUCCESS = "\tDownload successfully"
        self.DS_BASE_PCAP = "/Volumes/Archivio/malware"


    def __mta_download(self, url, dest, log_file):
        if not dest[-1] == '/':
            dest += '/'

        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)

        src = req.get(url).text
        soup = BeautifulSoup(src, 'lxml')

        name = soup.find('a', class_='menu_link')['href']

        # download pcapFile
        url_splited = url.split('/')
        pcap_url = '/'.join(url_splited[: len(url_splited) - 1]) + f'/{name}'
        dest_path = f'{dest + name}'

        if not os.path.isfile(dest_path) or not os.path.exists(dest_path):
            res = req.get(pcap_url)

            print(pcap_url)
            log_file.write(f'{datetime.datetime.now()} {pcap_url}\n')

            if res.status_code == 200:
                print(self.LOG_SUCCESS)
                log_file.write(f'{self.LOG_SUCCESS}\n')
                with open(dest_path, 'wb') as pcap:
                    pcap.write(res.content)
            else:
                print(self.LOG_FAIL)
                log_file.write(f'{self.LOG_FAIL}\n')
        else:
            print(f"{dest_path} - FILE GIÀ ESISTENTE")

    def __build_sample_attr(self, sample_name, scenario):
        if scenario == "original":
            return f"trazasOriginal/5GBdirectory/{sample_name}"
        else:
            return f"trazasNAT/{sample_name}"

    def __ha_download(self, sample_name, scenario, dest, log_file):
        if not dest[-1] == '/':
            dest += '/'

        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)

        url1 = "http://dataset.tlm.unavarra.es/ransomware/php/descargar.php?"  # usato per reperire il nome del file
        url2 = "http://dataset.tlm.unavarra.es/ransomware/downloads"  # usato per scaricare il file
        sample_value = self.__build_sample_attr(sample_name, scenario)

        payload = {'tokenDescarga': '',
                   'descargarTodo': '0',
                   'sample': sample_value}
        header = {'Cookie': 'PHPSESSID=eqbdqjkqrksd7u5l585mkk71t5',
                  'Origin': 'http://dataset.tlm.unavarra.es'}  # necessario altrimenti mi restituiva un contenuto vuoto

        post_res = req.post(url1, data=payload, headers=header)

        if post_res.status_code == 200:
            if len(os.listdir(dest)) == 0:
                post_res = post_res.json()['name'][12:]

                get_res = req.get(f'{url2 + post_res}')

                print(url2 + post_res)
                log_file.write(f'{datetime.datetime.now()} {url2 + post_res}\n')
                log_file.write(f'\tsample={sample_value}\n')

                if get_res.status_code == 200:
                    print(self.LOG_SUCCESS)
                    log_file.write(f'{self.LOG_SUCCESS}\n')
                    with open(f'{dest + post_res}', 'wb') as pcap:
                        pcap.write(get_res.content)
                else:
                    print(self.LOG_FAIL)
                    log_file.write(f'{self.LOG_FAIL}\n')
            else:
                print('File già esistente')
        else:
            print(self.LOG_NOTEXIST)
            log_file.write(f'{self.LOG_NOTEXIST}\n')

    def run(self):
        while True:
            sample = self.queue.get()

            if sample == "":
                break

            #path = f"raw/pcpa/{sample['Hash']}"
            path = f"{self.DS_BASE_PCAP}/{sample['Family']}/{sample['Hash']}"

            if "malware-traffic-analysis.net" in sample["Link"]:
                with open('download.log', 'a') as log:
                    self.__mta_download(sample["Link"], path, log)
            elif "hybrid-analysis.com" in sample["Link"]:
                with open('download.log', 'a') as log:
                    self.__ha_download(sample["SamplePcap"], sample["Scenario"], path, log)

            self.queue.task_done()

#------------------------------------------------------------
'''

elems = [{
            "SamplePcap": "samplesPcap_compress/cerber_03102016.pcap.gz",
            "Hash": "b17ff50e5abca3e850a433af42107314",
            "Link": "https://www.malware-traffic-analysis.net/2016/10/03/index2.html",
            "Scenario": "NAT",
            "Family": "Cerber"
        },
        {
            "SamplePcap": "samplesPcap_compress/Spora_17052017.pcap.gz",
            "Hash": "685c6834f8f81948822762cd13a48604",
            "Link": "https://www.malware-traffic-analysis.net/2017/05/17/index.html",
            "Scenario": "NAT",
            "Family": "Spora"
        }
        ]
samples = queue.Queue()
NO_WORKER = 2

for e in elems:
    samples.put(e)

# worker lavorano fino a che non ottengono una stinga vuota
for w in range(NO_WORKER):
    samples.put("")

# Create workers and add tot the queue
workers = []
for _ in range(NO_WORKER):
    worker = Downloader(samples)
    worker.start()
    workers.append(worker)
# Join workers to wait till they finished
for worker in workers:
    worker.join() 
'''
