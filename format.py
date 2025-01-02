import re
import os

date_prefixs = [
        #'',
        '2024-11-20',
        ]

path = './'
for exp in sorted(os.listdir(path)):
    if 'node' not in exp:
        continue
    #if '1nodes_rank0_train_70B_mbs2_bs128_tp8_pp1_cp1_iter6' not in exp:
    #    continue
    for fp in sorted(os.listdir(os.path.join(path,exp))):
        for date in sorted(os.listdir(os.path.join(path,exp,fp))):
            mark = False
            for date_prefix in date_prefixs:
                if date_prefix in date:
                    mark = True
            if not mark:
                continue
            _file = os.path.join(path,exp,fp,date,'output_perf.log')
            if os.path.exists(_file):
                _file = open(_file).readlines()
                result = time = 0
                for item in _file:
                    if 'TFLOP/s/GPU' in item:
                        _result = float(re.findall(r"GPU\): \d+\.?\d*",item)[0][5:])
                        if _result>result:
                            result = _result
                            time = re.findall(r"ms\): \d+\.?\d*",item)[0][5:]
                        #break
            else:
                result = 'not exist'
            if result == 0:
                continue
            print(exp, fp[10:], date, result, time)
