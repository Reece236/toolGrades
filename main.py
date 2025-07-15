import argparse
import train
import overall_model
import get_grades


def main():
    parser = argparse.ArgumentParser(description="MLB Tool Grades pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train tool grade models")
    subparsers.add_parser("overall", help="Train overall grade predictor")
    subparsers.add_parser("grades", help="Generate player grades")

    args, unknown = parser.parse_known_args()

    if args.command == "train":
        train.main()
    elif args.command == "overall":
        overall_model.main()
    elif args.command == "grades":
        get_grades.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
