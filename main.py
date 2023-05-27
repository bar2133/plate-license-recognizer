import argparse

from classes.Plate_license_recognition_system import PlateLicenseRecognitionSystem


def main():
    parser = argparse.ArgumentParser(
        prog='Plate License Recognition',
        description='the program recognize plate licenses in pictures or videos', )
    parser.add_argument('-t', '--data_type', type=str, choices=['video', 'picture'], required=True)
    parser.add_argument('-p', '--data_path', type=str, required=True, help='path to the data to process.')
    parser.add_argument('-pl', '--plate_license', type=str, required=False, nargs='*',
                        help='plate licenses to find in the frames.')
    parser.add_argument('-m', '--model_path', type=str, required=False,
                        help='path to model to use, if you want to replace the default.')
    parser.add_argument('-g', '--gpu', type=bool, required=False, help='use GPU or not',
                        default=True)
    parser.add_argument('-mr', '--match_ratio', type=int, required=False,
                        help='the ratio between the actual and the expected string.')
    args = vars(parser.parse_args())
    args = {key: args[key] for key in args if args[key] is not None}
    print(args)
    pls = PlateLicenseRecognitionSystem(**args)
    pls.find_plates(plates_license=args.get('plate_license', []))


if __name__ == '__main__':
    main()
