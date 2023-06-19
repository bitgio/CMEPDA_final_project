'''Assignment 1
Text statistics of a book from the Gutenberg Project
Final version
'''
import time
import argparse
import re
import string
import numpy as np
import matplotlib.pyplot as plt


t_initial = time.time()
obj = argparse.ArgumentParser()
obj.add_argument("file", type = argparse.FileType('r', encoding= 'UTF-8-sig'),
help = "This will be the book for the counting")
obj.add_argument("-s", "--skip",
help = "This skips the parts of the text which do not pertain to the book",
action="store_true")
obj.add_argument("-hi", "--histogram",
help = "This displays a histogram of the frequencies", action="store_true")
obj.add_argument("-c", "--characters",
help = "This print out the number of characters", action="store_true")
obj.add_argument("-w", "--words",
help = "This print out the number of words", action="store_true")
obj.add_argument("-l", "--lines",
help = "This print out the number of lines", action="store_true")
args = obj.parse_args()
il_file = args.file.read()
#Removing the index and the license
if args.skip:
    il_file = il_file[il_file.index('START OF THE PROJECT GUTENBERG'):
                      il_file.rfind('END OF THE PROJECT GUTENBERG')]
#Modifying the text file
il_file2 = len(re.findall(r'\w+', il_file))
il_file3 = len(il_file.splitlines())
il_file = il_file.replace('\n', '').replace(' ', '')
il_file = il_file.translate(str.maketrans('', '', string.punctuation))
#Counting each letter in lower case
count = {}
alf = list(string.ascii_lowercase)

for letter in alf:
    count[letter] = 0

COUNTER = 0
for letter in il_file:
    if letter.lower() in alf:
        count[letter.lower()] += 1
        COUNTER += 1

rel_freq = {k: ((v / COUNTER) * 100) for k, v in count.items()}

t_final = time.time()
elapsed_time = t_final - t_initial
print(f'The program lasts {elapsed_time}s to run.')
print(rel_freq)

if args.characters:
    print(f'The number of characters in the book is {COUNTER}.')
if args.words:
    print(f'The number of words in the book is {il_file2}.')
if args.lines:
    print(f'The number of lines in the book is {il_file3}.')
if args.histogram:
    fig, ax = plt.subplots()
    ax.bar(rel_freq.keys(), rel_freq.values(), width = 1, linewidth = 0.7)
    plt.xlabel('Letters of the English alphabet')
    plt.ylabel('Relative frequencies')
    ax.set(xticks=np.arange(0, 26))
    plt.show()
    