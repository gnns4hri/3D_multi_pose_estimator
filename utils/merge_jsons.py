import sys
import json
import datetime

total = 0

inputs = sys.argv[1:-1]
output = sys.argv[-1]

def wrong_usage():
    print(f'{sys.argv[0]} list.json of.json input.json files.json ....   output.json')
    sys.exit(-1)

if not output.endswith('.json'):
    wrong_usage()
for input_file in inputs:
    if not input_file.endswith('.json'):
        wrong_usage()

print('inputs', inputs)
print('oputputs', output)

all_data = []

for fname in inputs:
    print(fname)
    data = json.load(open(fname, 'r'))
    print(len(data))
    all_data += data

print('ALL DATA', len(all_data))
fd = open(output, 'w')
json.dump(all_data, fd)


