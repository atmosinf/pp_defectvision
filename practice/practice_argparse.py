import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='pracrtice argparse')

    parser.add_argument("--name", type=str, default="user", help="Name of the user")
    parser.add_argument("--age", type=int, default=18, help="Age of the user")
    parser.add_argument("--height", type=float, default=5.5, help="Height of the user in feet")
    parser.add_argument("--is-student", action='store_true', help="Is the user a student")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Name: {args.name}")
    print(f"Age: {args.age}")
    print(f"Height: {args.height} feet")
    print(f"Is Student: {args.is_student}")