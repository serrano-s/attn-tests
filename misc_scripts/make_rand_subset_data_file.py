import argparse
from random import shuffle
from random import random


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filename", type=str, required=True,
                        help="Local filename of data")
    parser.add_argument("--frac-to-take", type=float, required=True,
                        help="How much of the data to store in the new filename")
    parser.add_argument("--new-filename", type=str, required=True,
                        help="New filename for the data")

    parser.add_argument("--data-dir", required=False,
                        default="/homes/gws/sofias6/data/",
                        help="Base data dir")
    args = parser.parse_args()

    instances = []
    with open(args.data_dir + args.filename, 'r') as old_f:
        first_line = old_f.readline()
        for line in old_f:
            if line.strip() != '':
                instances.append(line)
        shuffle(instances)

    took = 0
    with open(args.data_dir + args.new_filename, 'w') as f:
        f.write(first_line)
        for instance in instances:
            decider = random()
            if decider < args.frac_to_take:
                f.write(instance)
                took += 1

    print("Wrote " + str(took) + " / " + str(len(instances)) + " instances to file " +
          str(args.data_dir + args.new_filename))


if __name__ == '__main__':
    main()
