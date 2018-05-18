import wave
import contextlib
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas
from collections import Counter

labels = {"Acoustic_guitar":0, "Applause":0, "Bark":0, "Bass_drum":0, "Burping_or_eructation":0, "Bus":0, "Cello":0, "Chime":0, "Clarinet":0, "Computer_keyboard":0, "Cough":0, "Cowbell":0, "Double_bass":0, "Drawer_open_or_close":0, "Electric_piano":0, "Fart":0, "Finger_snapping":0, "Fireworks":0, "Flute":0, "Glockenspiel":0, "Gong":0, "Gunshot_or_gunfire":0, "Harmonica":0, "Hi-hat":0, "Keys_jangling":0, "Knock":0, "Laughter":0, "Meow":0, "Microwave_oven":0, "Oboe":0, "Saxophone":0, "Scissors":0, "Shatter":0, "Snare_drum":0, "Squeak":0, "Tambourine":0, "Tearing":0, "Telephone":0, "Trumpet":0, "Violin_or_fiddle":0, "Writing":0}

labels_count=[]

verification = {'verified':0, 'non-verified':0}

reader = csv.reader(open('train.csv', 'r'))
names=[]
first_line=False
for row in reader:
    if first_line:
        name, label, verified = row
        names.append(name)
        labels[label]=labels[label]+1
        #labels_count.append(label)
        if verified=='1':
            verification['verified']=verification['verified']+1
        else:
            verification['non-verified']=verification['non-verified']+1
    else :
        first_line = True

print(labels)
df = pandas.DataFrame.from_dict(labels, orient='index')
plt.figure(1)
plt.title('Labels Distribution')
df.plot(kind='bar')

print(verification)

df = pandas.DataFrame.from_dict(verification, orient='index')
plt.figure(1)
plt.title('Manual verification Distribution')
df.plot(kind='bar')

durations=[]
max = 0
mean_duration=0
for fname in names:
    with contextlib.closing(wave.open("./audio_train/"+fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        mean_duration+=duration
        if duration > max:
            max=duration
        durations.append(np.floor(duration))

print(mean_duration/len(durations))

plt.figure(1)
plt.title('Durations')
plt.hist(durations,histtype='bar')
plt.show()

print(max)

#print(durations)
