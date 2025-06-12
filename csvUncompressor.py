import bz2

with bz2.open('data.csv.bz2', 'rt') as f_in, open('data.csv', 'w') as f_out:
    for line in f_in:
        f_out.write(line)
print("CSV file uncompressed successfully.")
