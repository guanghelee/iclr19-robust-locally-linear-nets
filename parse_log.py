import argparse
from utils import print_args
from collections import defaultdict

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Parse pytorch records')
    parser.add_argument('--file-list', type=str,
                        help='the file name for the list of ')
    args = parser.parse_args()
    print_args(args)

    best_model = defaultdict(list)

    with open(args.file_list) as list_fp:
        for line in list_fp:
            fn = line.strip()
            count = 0
            print_str = []
            with open(fn) as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) > 5 and line[:5] == 'valid':
                        count = 3
                        print_str = []
                    if count != 0:
                        count -= 1
                        if count == 2:
                            line = line[-4:-1]
                        elif count == 1:
                            line = line.split(' ')[-1]
                        else:
                            line = line.split(',')[-3]
                        print_str.append(line)

                print_str[2] = float(print_str[2])
                mid_dist = print_str[2]
                acc = (print_str[0])
                if len(best_model[acc]) == 0 or best_model[acc][-1] < mid_dist:
                    best_model[acc] = [fn] + print_str

                print(fn, print_str)

    print()
    for acc in sorted(best_model, reverse=True):
        print('acc = {}, # of region = {}, mid dist = {:.4f}, fn = {}'.format(best_model[acc][1], best_model[acc][2], best_model[acc][3], best_model[acc][0]))

if __name__ == '__main__':
    main()
