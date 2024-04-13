import os


def main(input_dir, output_dir):
    for root, dirs, files in os.walk("org_data"):
        for dir in dirs:
            


if __name__ == "__main__":
    input_dir = "org_data"
    output_dir = "new_data"
    main()