import random

def generate_dataset(filename, num_samples):
    with open(filename, "w") as file:
        for _ in range(num_samples):
            input_array = [round(random.uniform(-1000, 1000), 2) for _ in range(10)]
            output_array = sorted(input_array)
            file.write(f"[{input_array}, {output_array}]\n")

    print(f"Dataset with {num_samples} samples saved to '{filename}'.")


# This is a quick hack to solve a tokenization problem
def modify_dataset(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            modified_line = line.replace(' [', ' [ ')
            outfile.write(modified_line)


PATH = "sorting_dataset_10m.txt"
generate_dataset(PATH, 10_000_000)
input_path = 'sorting_dataset_10m.txt'  # Replace with your input file path
output_path = 'sorting_moded_dataset_10m.txt'  # Replace with your desired output file path
modify_dataset(input_path, output_path)
