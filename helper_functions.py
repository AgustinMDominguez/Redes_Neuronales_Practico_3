
from json import loads

def lineprint(msg='', ch = '-'):
    for _ in range(80):
        print(ch, end='')
    print('\n\t' + msg)

def myprint(st):
    print("\t"+st)

def loadTrainingLog(n_hidden, mode):
    with open(f"training_logs/epch40_h{n_hidden}_mode{mode}_traintest.txt", 'r') as f:
        st = f.read()
        return loads(st)
