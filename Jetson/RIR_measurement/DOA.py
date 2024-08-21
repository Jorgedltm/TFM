# En este script lo que haremos será pasar como argumento el DOA real y medirlo 15 veces sacando una media del estimado para así guardarlo en un archivo csv.
from  usb_4_mic_array.tuning import Tuning
import usb.core
import usb.util
import time
import pandas as pd
import sys
import numpy as np

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if dev:

    Mic_tuning = Tuning(dev)
    direction = np.array([])
    counter = 0
    print(Mic_tuning.direction)
    direction = np.append(direction,Mic_tuning.direction)
    counter += 1
    while True:
        try:
            print(Mic_tuning.direction)
            time.sleep(1)
            if Mic_tuning.direction>180:
               dir = 360 - Mic_tuning.direction
            else:
               dir = Mic_tuning.direction
            counter += 1
            direction = np.append(direction, dir)
            if counter>=15:
               raise KeyboardInterrupt 
        except KeyboardInterrupt:
            print(f"La DOA es la siguiente:{direction}.")
            DOA_med_csv= '../TFM/DOA_medidas.csv'
            df = pd.read_csv(DOA_med_csv,decimal=',',sep=';')
            new_row_data ={'Real_DOA': int(sys.argv[1]),'Mean_measured_DOA': float(np.mean(direction)),'Std_measured_DOA': float(np.std(direction))} 
            df = df.append(new_row_data, ignore_index=True)
            df.to_csv(DOA_med_csv, index=False, decimal=',',sep=';')
            break
