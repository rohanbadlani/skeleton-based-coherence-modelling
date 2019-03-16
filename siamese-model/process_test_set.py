import csv
import sys
import pdb

def process(input_file, output_file):
    with open(input_file,'r') as tsvin, open(output_file, 'w') as csvout:
        tsvin = csv.reader(tsvin, delimiter='\t')
        csvout = csv.writer(csvout, delimiter='\t')
        for row in tsvin:
            output_row = [row[2], row[0], row[1]]
            csvout.writerow(output_row)

if(len(sys.argv)!=3):
    print("process_test_set usage: python process_test_set.py <input-filepath> <output-filepath>")
    sys.exit(1)

input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

process(input_file, output_file)